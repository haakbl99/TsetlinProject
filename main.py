import argparse
import logging
import os
import pickle
from time import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as tm


_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REMOVE = ("headers", "footers", "quotes")

PAIR_CLASSES = ["different", "same"]


def preprocess(texts, lemmatize, keep_stopwords):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 2_000_000
    documents = []
    for doc in nlp.pipe(texts, batch_size=200):
        tokens = []
        for t in doc:
            if not t.is_alpha or len(t) < 2:
                continue
            if not keep_stopwords and t.is_stop:
                continue
            tokens.append(t.lemma_.lower() if lemmatize else t.lower_)
        documents.append(tokens)
    return documents


def load_tokens(subset, texts, args):
    os.makedirs(args.cache_dir, exist_ok=True)
    cats = "all" if not args.categories else "-".join(sorted(args.categories))
    name = f"{subset}_{cats}_{'lemma' if args.lemmatize else 'lower'}_{'stop' if args.keep_stopwords else 'nostop'}.pkl"
    path = os.path.join(args.cache_dir, name)
    if os.path.exists(path):
        _LOGGER.info(f"Loading cached tokens from {path}")
        with open(path, "rb") as f:
            documents = pickle.load(f)
    else:
        _LOGGER.info(f"Processing {len(texts)} {subset} posts with spaCy (this takes a few minutes)...")
        documents = preprocess(texts, args.lemmatize, args.keep_stopwords)
        with open(path, "wb") as f:
            pickle.dump(documents, f)
    if len(documents) != len(texts):
        raise RuntimeError(f"{path} holds {len(documents)} posts but {len(texts)} were loaded. "
                           f"Delete the cache folder ({args.cache_dir}) and run again.")
    return documents


def bool_flag(parser, name, default, help_text):
    parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)



def make_pairs(X, y, n_pairs, rng):
    by_class = [np.where(y == c)[0] for c in np.unique(y)]
    by_class = [idx for idx in by_class if len(idx) >= 2]
    X_pairs = np.zeros((n_pairs, 2 * X.shape[1]), dtype=np.uint8)
    Y_pairs = rng.integers(0, 2, n_pairs).astype(np.uint32)
    pair_index = np.zeros((n_pairs, 2), dtype=np.int64)
    for i in range(n_pairs):
        if Y_pairs[i] == 1:
            c = rng.integers(len(by_class))
            a, b = rng.choice(by_class[c], 2, replace=False)
        else:
            c1, c2 = rng.choice(len(by_class), 2, replace=False)
            a, b = rng.choice(by_class[c1]), rng.choice(by_class[c2])
        X_pairs[i, :X.shape[1]] = X[a]
        X_pairs[i, X.shape[1]:] = X[b]
        pair_index[i] = (a, b)
    return X_pairs, Y_pairs, pair_index


def pair_feature_names(feature_names, selected_features):
    words = [feature_names[k] for k in selected_features]
    return ["A:" + w for w in words] + ["B:" + w for w in words]


def get_feature_maps(class_tm):
    fmap = class_tm.map_ta_id_to_feature_id()
    half = class_tm.number_of_literals_per_leaf // 2
    negated = np.zeros(class_tm.number_of_literals_per_leaf, dtype=bool)
    if class_tm.append_negated:
        negated[half:] = True
    return fmap, negated



def get_clause_weights(class_tm):
    import pycuda.driver as cuda
    weights = np.empty(class_tm.number_of_outputs * class_tm.number_of_clauses, dtype=np.int32)
    cuda.memcpy_dtoh(weights, class_tm.clause_weights_gpu)
    return weights[:class_tm.number_of_clauses]


def literal_profile(class_tm, names, top_n):
    literals = class_tm.get_literals()
    weights = get_clause_weights(class_tm)
    fmap, negated = get_feature_maps(class_tm)
    groups, negated_share = {}, {}
    for title, positive, neg in [
        ("Positive polarity (presence supports the class)", True, False),
        ("Positive polarity (absence supports the class)", True, True),
        ("Negative polarity (presence counts against the class)", False, False),
        ("Negative polarity (absence counts against the class)", False, True),
    ]:
        importance = np.zeros(len(names))
        clause_mask = weights > 0 if positive else weights < 0
        for c in np.where(clause_mask)[0]:
            leaf, ta = np.nonzero(literals[c])
            keep = negated[ta] == neg
            np.add.at(importance, fmap[leaf[keep], ta[keep]], abs(weights[c]))
        order = np.argsort(-importance)[:top_n]
        words = [names[k] for k in order if importance[k] > 0]
        groups[title] = ["¬'" + w + "'" for w in words] if neg else words
    for label, mask in [("positive", weights > 0), ("negative", weights < 0)]:
        included = literals[mask]
        total = included.sum()
        negated_share[label] = included[..., negated].sum() / total if total else 0.0
    return groups, negated_share, literals, weights

def render_clause(literals_c, fmap, negated, names, max_literals=12):
    tas = np.nonzero(literals_c[0])[0]
    if len(tas) == 0:
        return "(empty)"
    tas = sorted(tas, key=lambda t: negated[t])
    parts = [("¬" if negated[t] else "") + names[fmap[0, t]] for t in tas]
    extra = f" ∧ …+{len(parts) - max_literals}" if len(parts) > max_literals else ""
    return " ∧ ".join(parts[:max_literals]) + extra


def print_class_explanation(class_tm, name, names, args):
    groups, negated_share, literals, weights = literal_profile(class_tm, names, args.profile_size)
    print(f"\n=== {name} ===")
    print(f"  Share of included literals that are negated: "
          f"{100 * negated_share['positive']:.1f}% in positive clauses, "
          f"{100 * negated_share['negative']:.1f}% in negative clauses")
    for title, words in groups.items():
        print(f"  {title}: {' '.join(words) if words else '(none)'}")
    fmap, negated = get_feature_maps(class_tm)
    sizes = literals.reshape(literals.shape[0], -1).sum(axis=1)
    for polarity, mask in [("for", weights > 0), ("against", weights < 0)]:
        candidates = [c for c in np.where(mask)[0] if sizes[c] > 0]
        candidates.sort(key=lambda c: (-abs(weights[c]), sizes[c]))
        for c in candidates[:args.clauses_to_print]:
            print(f"  Clause #{c} (weight {weights[c]}, votes {polarity}): "
                  f"{render_clause(literals[c], fmap, negated, names)}")


def cosine_scores(tfidf_docs, pair_index):
    a = tfidf_docs[pair_index[:, 0]]
    b = tfidf_docs[pair_index[:, 1]]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


if __name__ == "__main__":
    # ENTRY POINT
    # The code below runs only when this file is executed directly, not when imported.

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--number-of-clauses", default=2000, type=int)
    parser.add_argument("--number-of-state-bits", default=16, type=int)
    parser.add_argument("--T", default=20000, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--constant-update-p", action="store_true")
    parser.add_argument("--binary-inference", action="store_true")
    parser.add_argument("--and-group-normalization", action="store_true")
    parser.add_argument("--no-clipping", action="store_true")
    parser.add_argument("--boost-true-positive-feedback", default=1, type=int)
    bool_flag(parser, "weighted-clauses", False, "let clauses have different voting weights")
    bool_flag(parser, "append-negated", True, "also give the model negated words (needed for reasoning by elimination)")
    parser.add_argument("--max-ngram", default=1, type=int)
    parser.add_argument("--min-df", default=2, type=int)
    parser.add_argument("--features", default=10000, type=int, help="words per document (the pair has twice as many)")
    bool_flag(parser, "lemmatize", True, "reduce words to their lemma with spaCy")
    bool_flag(parser, "keep-stopwords", False, "keep words such as 'the' and 'not'")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="newsgroups to use, e.g. sci.crypt sci.med rec.autos (default: all 20)")
    parser.add_argument("--train-pairs", default=20000, type=int)
    parser.add_argument("--val-pairs", default=2000, type=int)
    parser.add_argument("--test-pairs", default=5000, type=int)
    parser.add_argument("--val-fraction", default=0.1, type=float,
                        help="share of training posts kept for validation pairs")
    bool_flag(parser, "baseline", True, "also run a TF-IDF cosine-similarity baseline")
    bool_flag(parser, "explain", True, "print the learned literals and example clauses")
    parser.add_argument("--profile-size", default=20, type=int)
    parser.add_argument("--clauses-to-print", default=3, type=int)
    parser.add_argument("--save-model", default=None, type=str, help="file to save the last run's model to")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cache-dir", default=".cache_20newsgroups", type=str)
    args = parser.parse_args()

    if args.categories:
        all_names = fetch_20newsgroups(subset="train", remove=REMOVE).target_names
        unknown = [c for c in args.categories if c not in all_names]
        if unknown:
            parser.error(f"Unknown categories: {', '.join(unknown)}. Valid names: {', '.join(all_names)}")
        if len(args.categories) < 2:
            parser.error("Choose at least two categories, so that 'different' pairs exist.")

    _LOGGER.info("Loading 20 Newsgroups...")
    train = fetch_20newsgroups(subset="train", remove=REMOVE, categories=args.categories)
    test = fetch_20newsgroups(subset="test", remove=REMOVE, categories=args.categories)
    category_names = train.target_names
    _LOGGER.info(f"Loaded {len(train.data)} training and {len(test.data)} test posts, "
                 f"{len(category_names)} categories")

    training_documents = load_tokens("train", train.data, args)
    testing_documents = load_tokens("test", test.data, args)
    cat_train_all = train.target
    cat_test = test.target


    idx_train, idx_val = train_test_split(np.arange(len(training_documents)), test_size=args.val_fraction,
                                          stratify=cat_train_all, random_state=args.seed)
    train_docs = [training_documents[i] for i in idx_train]
    val_docs = [training_documents[i] for i in idx_val]
    cat_train = cat_train_all[idx_train]
    cat_val = cat_train_all[idx_val]
    _LOGGER.info(f"Documents: {len(train_docs)} for training pairs, {len(val_docs)} for validation pairs, "
                 f"{len(testing_documents)} for test pairs")


    vectorizer_X = CountVectorizer(tokenizer=lambda s: s, token_pattern=None, ngram_range=(1, args.max_ngram),
                                   lowercase=False, binary=True, min_df=args.min_df)
    X_train_docs = vectorizer_X.fit_transform(train_docs)
    X_val_docs = vectorizer_X.transform(val_docs)
    X_test_docs = vectorizer_X.transform(testing_documents)
    feature_names = vectorizer_X.get_feature_names_out()
    _LOGGER.info(f"Vocabulary size: {len(feature_names)} n-grams")

    n_features = min(args.features, X_train_docs.shape[1])
    SKB = SelectKBest(chi2, k=n_features)
    SKB.fit(X_train_docs, cat_train)
    selected_features = SKB.get_support(indices=True)
    X_train_docs = SKB.transform(X_train_docs).toarray().astype(np.uint8)
    X_val_docs = SKB.transform(X_val_docs).toarray().astype(np.uint8)
    X_test_docs = SKB.transform(X_test_docs).toarray().astype(np.uint8)
    names = pair_feature_names(feature_names, selected_features)
    _LOGGER.info(f"Documents with no selected words: {int((X_train_docs.sum(axis=1) == 0).sum())} train, "
                 f"{int((X_test_docs.sum(axis=1) == 0).sum())} test")

    rng = np.random.default_rng(args.seed)
    X_train, Y_train, _ = make_pairs(X_train_docs, cat_train, args.train_pairs, rng)
    X_val, Y_val, val_index = make_pairs(X_val_docs, cat_val, args.val_pairs, rng)
    X_test, Y_test, test_index = make_pairs(X_test_docs, cat_test, args.test_pairs, rng)
    _LOGGER.info(f"Pairs: {len(Y_train)} train, {len(Y_val)} validation, {len(Y_test)} test, "
                 f"{2 * n_features} features each, {100 * Y_train.mean():.1f}% 'same' in training")


    if args.baseline:
        tfidf = TfidfVectorizer(tokenizer=lambda s: s, token_pattern=None, lowercase=False,
                                min_df=args.min_df, sublinear_tf=True)
        tfidf.fit(train_docs)
        val_scores = cosine_scores(tfidf.transform(val_docs), val_index)
        test_scores = cosine_scores(tfidf.transform(testing_documents), test_index)
        thresholds = np.unique(val_scores)
        best_t = max(thresholds, key=lambda t: ((val_scores >= t) == Y_val).mean())
        base_pred = (test_scores >= best_t).astype(np.uint32)
        _LOGGER.info(f"Baseline cosine similarity (threshold {best_t:.3f}) test accuracy: "
                     f"{100 * accuracy_score(Y_test, base_pred):.2f}")

    clauses = args.number_of_clauses
    structure = ((tm.AND_GROUP, 2 * n_features), (tm.AND_GROUP, 1))
    literals = 2 * (2 * n_features) if args.append_negated else 2 * n_features
    estimate = clauses * ((literals - 1) // 32 + 1) * args.number_of_state_bits * 4 * 2
    _LOGGER.info(f"Estimated GPU memory for automata: {estimate / 1024 ** 2:.0f} MB (lower bound)")


    stats_name = (f"newsgroups_pairs_statistics_{clauses}_{args.T}_{args.s:.2f}_{args.number_of_state_bits}_"
                  f"{n_features}_{args.train_pairs}.txt")
    f = open(stats_name, "w")
    f.write("run epoch validation_accuracy\n")

    test_results = []
    for r in range(args.runs):
        seed = int(rng.integers(1, 10000))
        np.random.seed(seed)
        model = MultiClassTsetlinMachine(
            clauses,
            args.T,
            args.s,
            binary_inference=args.binary_inference,
            constant_update_p=args.constant_update_p,
            and_group_normalization=args.and_group_normalization,
            weighted_clauses=args.weighted_clauses,
            seed=seed,
            number_of_state_bits=args.number_of_state_bits,
            boost_true_positive_feedback=args.boost_true_positive_feedback,
            no_clipping=args.no_clipping,
            append_negated=args.append_negated,
            hierarchy_structure=structure
        )


        _LOGGER.info(f"Run {r + 1}/{args.runs} (seed {seed}): {args.epochs} epochs")
        best_epoch, best_val = 0, -1.0
        for e in range(args.epochs):
            start_training = time()
            model.fit(X_train, Y_train)
            stop_training = time()

            start_validation = time()
            result = 100 * (model.predict(X_val) == Y_val).mean()
            stop_validation = time()
            if result > best_val:
                best_epoch, best_val = e + 1, result
            _LOGGER.info(f"#{r + 1}/{e + 1} Validation accuracy: {result:.2f}% "
                         f"Training: {stop_training - start_training:.2f}s Validation: {stop_validation - start_validation:.2f}s")
            f.write(f"{r} {e} {result:.2f}\n")
            f.flush()


        Y_pred = model.predict(X_test)
        test_acc = 100 * accuracy_score(Y_test, Y_pred)
        test_f1 = 100 * f1_score(Y_test, Y_pred, average="macro")
        test_results.append((test_acc, test_f1))
        _LOGGER.info(f"Run {r + 1}: best validation accuracy {best_val:.2f}% at epoch {best_epoch}")
        _LOGGER.info(f"Run {r + 1}: test accuracy {test_acc:.2f}%, macro-F1 {test_f1:.2f}%")
        f.write(f"# run {r} test_accuracy {test_acc:.2f} test_macro_f1 {test_f1:.2f}\n")
        f.flush()

    f.close()

    if args.runs > 1:
        accs = np.array([a for a, _ in test_results])
        f1s = np.array([b for _, b in test_results])
        print(f"\nTest accuracy over {args.runs} runs: {accs.mean():.2f} ± {accs.std():.2f}")
        print(f"Test macro-F1 over {args.runs} runs: {f1s.mean():.2f} ± {f1s.std():.2f}")

    print(f"\nLast run - test accuracy: {test_acc:.2f}, macro-F1: {test_f1:.2f}\n")
    print(classification_report(Y_test, Y_pred, target_names=PAIR_CLASSES, digits=3))
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(f"               different   same")
    print(f"  different   {cm[0, 0]:9d}  {cm[0, 1]:5d}")
    print(f"  same        {cm[1, 0]:9d}  {cm[1, 1]:5d}")

    test_cat_a = cat_test[test_index[:, 0]]
    same_mask = Y_test == 1
    print("\nTest accuracy on 'same' pairs by newsgroup:")
    for c, name in enumerate(category_names):
        m = same_mask & (test_cat_a == c)
        if m.sum():
            print(f"  {name:28s} {100 * (Y_pred[m] == 1).mean():6.2f}%  ({int(m.sum())} pairs)")


    if args.save_model:
        with open(args.save_model, "wb") as mf:
            pickle.dump({"model": model.save(), "names": names, "category_names": category_names}, mf)
        _LOGGER.info(f"Saved the last run's model to {args.save_model}")
    # SAVE THE MODEL (OPTIONAL)
    # Uses the library's save() function and stores the column names alongside it, so the
    # clauses can be explained later without retraining.

    if args.explain:
        print("\nLearned literals and example clauses (last run):")
        for class_id, name in enumerate(PAIR_CLASSES):
            print_class_explanation(model.tms[class_id], name, names, args)
