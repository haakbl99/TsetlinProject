import argparse
import logging
import os
import pickle
import re
from time import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import PyHierarchicalTsetlinMachineCUDA.tm as tm
# IMPORTS
# - scikit-learn supplies the 20 Newsgroups dataset, the vectorizers, chi-squared
#   feature selection, the validation split and the evaluation metrics.
# - PyHierarchicalTsetlinMachineCUDA supplies the Tsetlin Machine, as in the multi-concept
#   sample. It runs only on an NVIDIA GPU through pycuda. The "tm" module holds the
#   hierarchy building blocks (tm.AND_GROUP and so on).
# - re splits the posts into words, and scikit-learn's ENGLISH_STOP_WORDS is the stop-word list.
# - NLTK's Snowball stemmer is imported later, inside preprocess(), and only when posts need to
#   be processed. It replaces spaCy, which is no longer needed.

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# LOGGING
# Creates a logger for this file and shows messages at INFO level and above.

REMOVE = ("headers", "footers", "quotes")
# METADATA TO STRIP
# Removes email headers, signatures and quoted replies from every post. Without this,
# two posts could look "the same" because they quote each other or share a signature,
# rather than because they share a topic.

PAIR_CLASSES = ["different", "same"]
# LABEL NAMES
# Y=0 means the two documents come from different newsgroups, and Y=1 means the same newsgroup.


WORD = re.compile(r"[a-z]+")


def preprocess(texts, stem, keep_stopwords):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    stems = {}
    documents = []
    for text in texts:
        tokens = []
        for word in WORD.findall(text.lower()):
            if len(word) < 2:
                continue
            if not keep_stopwords and word in ENGLISH_STOP_WORDS:
                continue
            if stem:
                if word not in stems:
                    stems[word] = stemmer.stem(word)
                word = stems[word]
            tokens.append(word)
        documents.append(tokens)
    return documents
# TOKENIZE AND STEM THE POSTS
# Replaces the spaCy lemmatizer with a stemmer:
# - Each post is lowercased and split into runs of letters, so numbers and punctuation are dropped.
#   Words of one letter are skipped.
# - Stop words such as "the" and "and" are removed (before stemming) unless --keep-stopwords is set.
# - The Snowball ("Porter2") stemmer cuts each word down to its stem by rule, for example
#   "encryption" and "encrypted" -> "encrypt", "universities" -> "univers". Unlike a lemmatizer it
#   does not look the word up or use its context, so stems are not always real words, and it
#   can merge words a lemmatizer would keep apart. In return it is far faster: seconds instead of
#   minutes for all posts.
# - Each distinct word is stemmed once and remembered in "stems", which speeds things up further.
# - With --no-stem, words are only lowercased, using the same splitting, so a comparison
#   measures only the effect of stemming.


def load_tokens(subset, texts, args):
    os.makedirs(args.cache_dir, exist_ok=True)
    cats = "all" if not args.categories else "-".join(sorted(args.categories))
    name = f"{subset}_{cats}_re-{'stem' if args.stem else 'lower'}_{'stop' if args.keep_stopwords else 'nostop'}.pkl"
    path = os.path.join(args.cache_dir, name)
    if os.path.exists(path):
        _LOGGER.info(f"Loading cached tokens from {path}")
        with open(path, "rb") as f:
            documents = pickle.load(f)
    else:
        _LOGGER.info(f"Processing {len(texts)} {subset} posts ({'stemming' if args.stem else 'no stemming'})...")
        documents = preprocess(texts, args.stem, args.keep_stopwords)
        with open(path, "wb") as f:
            pickle.dump(documents, f)
    if len(documents) != len(texts):
        raise RuntimeError(f"{path} holds {len(documents)} posts but {len(texts)} were loaded. "
                           f"Delete the cache folder ({args.cache_dir}) and run again.")
    return documents
# CACHE THE PROCESSED POSTS
# The processed tokens are saved the first time and reloaded later. Stemming is fast, so this
# matters less than it did with spaCy, but it still saves a few seconds per run.
# Each combination of categories and preprocessing settings has its own file, and the length
# check stops the script if a cache file does not match the loaded posts.
# The file names start with "re-" for the new tokenizer, so the old spaCy cache files are never
# loaded by mistake. You can delete those old files.


def bool_flag(parser, name, default, help_text):
    parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)
# TRUE/FALSE SETTINGS
# Creates a pair of flags such as --stem and --no-stem.


ENCODINGS = ("bundle", "concat")


def combine(a, b, encoding):
    if encoding == "bundle":
        return a | b
    return np.concatenate([a, b])
# COMBINE TWO DOCUMENT VECTORS INTO ONE PAIR VECTOR
# a and b are the two documents' boolean vectors: one value per selected word, 1 if the
# document contains it.
# - bundle (OR): 1 if EITHER document contains the word, which is the union of the two word sets.
#   In hyperdimensional computing, bundling usually means a majority vote over the vectors, but
#   with only two vectors the vote is tied whenever they disagree; OR settles every tie as 1,
#   the usual choice for sets. The result has one column per word, the same length as a single
#   document, and the order of the documents does not matter: combine(a, b) == combine(b, a).
#   It cannot tell whether one document or both contain a word.
# - concat: the two vectors side by side (2 columns per word), the original encoding, kept for
#   comparison. It keeps track of which document contains which word.


def make_pairs(X, y, n_pairs, rng, encoding):
    by_class = [np.where(y == c)[0] for c in np.unique(y)]
    by_class = [idx for idx in by_class if len(idx) >= 2]
    width = 2 * X.shape[1] if encoding == "concat" else X.shape[1]
    X_pairs = np.zeros((n_pairs, width), dtype=np.uint8)
    Y_pairs = rng.integers(0, 2, n_pairs).astype(np.uint32)
    pair_index = np.zeros((n_pairs, 2), dtype=np.int64)
    for i in range(n_pairs):
        if Y_pairs[i] == 1:
            c = rng.integers(len(by_class))
            a, b = rng.choice(by_class[c], 2, replace=False)
        else:
            c1, c2 = rng.choice(len(by_class), 2, replace=False)
            a, b = rng.choice(by_class[c1]), rng.choice(by_class[c2])
        X_pairs[i] = combine(X[a], X[b], encoding)
        pair_index[i] = (a, b)
    return X_pairs, Y_pairs, pair_index
# BUILD DOCUMENT PAIRS (THE TARGET FROM THE ASSIGNMENT)
# For each example, the label is chosen first with a coin flip, so about half the pairs are
# "same" and half are "different":
# - Y=1: pick a newsgroup at random and draw two different documents from it.
# - Y=0: pick two different newsgroups and draw one document from each.
# Drawing two documents fully at random would give "different" about 19 times in 20, so a
# model that always said "different" would score about 95%. Choosing the label first avoids that.
# The two documents' boolean word vectors are then combined with combine(), using the chosen
# encoding. The pairs are drawn in exactly the same way for every encoding, so with the same
# seed all encodings are trained and tested on the same document pairs.
# pair_index records which documents were used, for the similarity baseline below.
# Pairs are stored as uint8 to save memory. The library converts them to uint32 itself.


def pair_feature_names(feature_names, selected_features, encoding):
    words = [feature_names[k] for k in selected_features]
    if encoding == "bundle":
        return ["OR:" + w for w in words]
    return ["A:" + w for w in words] + ["B:" + w for w in words]
# NAMES FOR THE PAIR COLUMNS
# Labels each column so the explanations say what a literal means:
# - bundle: "OR:hockey" means at least one document contains "hockey", and "¬OR:hockey" means
#   neither does. That negated literal is exactly the shared-absence test elimination rules use.
# - concat: "A:hockey" and "B:hockey" refer to each document separately, as before.


def get_feature_maps(class_tm):
    fmap = class_tm.map_ta_id_to_feature_id()
    half = class_tm.number_of_literals_per_leaf // 2
    negated = np.zeros(class_tm.number_of_literals_per_leaf, dtype=bool)
    if class_tm.append_negated:
        negated[half:] = True
    return fmap, negated
# MAP AUTOMATA TO COLUMNS
# map_ta_id_to_feature_id() from the library returns the input column each automaton refers to.
# With append_negated, the first half of the automata are the plain features and the second
# half are the negated ones (¬feature).


def get_clause_weights(class_tm):
    import pycuda.driver as cuda
    weights = np.empty(class_tm.number_of_outputs * class_tm.number_of_clauses, dtype=np.int32)
    cuda.memcpy_dtoh(weights, class_tm.clause_weights_gpu)
    return weights[:class_tm.number_of_clauses]
# READ THE CLAUSE WEIGHTS
# Copies every clause's weight from the GPU in one step. Newer versions of the library have a
# get_clause_weight() function, but the PyPI release (0.3.3) does not, so this reads the weights
# the same way the library's own save() function does. It works with both versions.
# Each class's model has a single output, so the first number_of_clauses values are its weights.


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
# LITERAL IMPORTANCE FOR ONE CLASS
# This library has no literal_importance() function, so the four groups are rebuilt from the
# automata. A column's importance is the number of times it is included across that polarity's
# clauses, weighted by the size of each clause's weight. Positive-weight clauses vote for the
# class and negative-weight clauses vote against it.
# negated_share is the fraction of included literals that are negated. A high share means the
# model reasons mostly by elimination: it relies on which words are missing.


def render_clause(literals_c, fmap, negated, names, max_literals=12):
    tas = np.nonzero(literals_c[0])[0]
    if len(tas) == 0:
        return "(empty)"
    tas = sorted(tas, key=lambda t: negated[t])
    parts = [("¬" if negated[t] else "") + names[fmap[0, t]] for t in tas]
    extra = f" ∧ …+{len(parts) - max_literals}" if len(parts) > max_literals else ""
    return " ∧ ".join(parts[:max_literals]) + extra
# WRITE A CLAUSE IN WORDS
# A vanilla clause is a single AND of literals. This writes it with real words instead of the
# column numbers (x123) that the library's print_hierarchy() uses. Plain literals come first,
# then negated ones, and the clause is cut off after max_literals.


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
# EXPLANATION FOR ONE CLASS
# Prints the share of negated literals, the four literal groups, and example clauses in words:
# up to clauses_to_print clauses voting for the class and the same number voting against it.
# Shorter clauses are shown first among equal weights because they are easier to read.


def cosine_scores(tfidf_docs, pair_index):
    a = tfidf_docs[pair_index[:, 0]]
    b = tfidf_docs[pair_index[:, 1]]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()
# COSINE SIMILARITY OF EACH PAIR
# TF-IDF vectors are already scaled to length 1, so the sum of their element-wise product is
# the cosine similarity: close to 1 for documents with very similar words, and 0 for no overlap.


if __name__ == "__main__":
    # ENTRY POINT
    # The code below runs only when this file is executed directly, not when imported.

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--number-of-clauses", default=1000, type=int)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--T", default=500, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--constant-update-p", action="store_true")
    parser.add_argument("--binary-inference", action="store_true")
    parser.add_argument("--and-group-normalization", action="store_true")
    parser.add_argument("--no-clipping", action="store_true")
    parser.add_argument("--boost-true-positive-feedback", default=1, type=int)
    bool_flag(parser, "weighted-clauses", True, "let clauses have different voting weights")
    bool_flag(parser, "append-negated", True, "also give the model negated words (needed for reasoning by elimination)")
    parser.add_argument("--max-ngram", default=1, type=int)
    parser.add_argument("--min-df", default=2, type=int)
    parser.add_argument("--features", default=500, type=int, help="words in the lexicon (selected features)")
    parser.add_argument("--pair-encoding", default="bundle", choices=ENCODINGS,
                        help="how two documents are combined: bundle (OR) or concat (side by side)")
    bool_flag(parser, "stem", True, "reduce words to their stem with the Snowball stemmer")
    bool_flag(parser, "keep-stopwords", False, "keep words such as 'the' and 'not'")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="newsgroups to use, e.g. sci.crypt sci.med rec.autos (default: all 20)")
    parser.add_argument("--train-pairs", default=50000, type=int)
    parser.add_argument("--val-pairs", default=2000, type=int)
    parser.add_argument("--test-pairs", default=5000, type=int)
    parser.add_argument("--val-fraction", default=0.1, type=float, help="share of training posts kept for validation pairs")
    bool_flag(parser, "baseline", True, "also run a TF-IDF cosine-similarity baseline")
    bool_flag(parser, "explain", True, "print the learned literals and example clauses")
    parser.add_argument("--profile-size", default=20, type=int)
    parser.add_argument("--clauses-to-print", default=3, type=int)
    parser.add_argument("--save-model", default="model.pkl", type=str,
                        help="file to save the last run's model to (empty string to skip)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cache-dir", default="/home/hufsa/TsetlinProject/TsetlinProject/.cache_20newsgroups", type=str)
    args = parser.parse_args()
   

    if args.categories:
        all_names = fetch_20newsgroups(subset="train", remove=REMOVE).target_names
        unknown = [c for c in args.categories if c not in all_names]
        if unknown:
            parser.error(f"Unknown categories: {', '.join(unknown)}. Valid names: {', '.join(all_names)}")
        if len(args.categories) < 2:
            parser.error("Choose at least two categories, so that 'different' pairs exist.")
    # CHECK THE CATEGORY NAMES
    # Stops with a clear message if a category name is misspelled. At least two categories are
    # needed, because a "different" pair needs two newsgroups.

    _LOGGER.info("Loading 20 Newsgroups...")
    train = fetch_20newsgroups(subset="train", remove=REMOVE, categories=args.categories)
    test = fetch_20newsgroups(subset="test", remove=REMOVE, categories=args.categories)
    category_names = train.target_names
    _LOGGER.info(f"Loaded {len(train.data)} training and {len(test.data)} test posts, "
                 f"{len(category_names)} categories")
    # LOAD THE DATA
    # The dataset holds about 18,800 posts in total after duplicates were removed, split by date
    # into a training set and a newer test set. The category labels are used to build pairs.

    training_documents = load_tokens("train", train.data, args)
    testing_documents = load_tokens("test", test.data, args)
    cat_train_all = train.target
    cat_test = test.target
    # PREPARE THE TOKENS AND CATEGORY LABELS
    # Runs preprocess() on both sets, or loads the cached result.

    idx_train, idx_val = train_test_split(np.arange(len(training_documents)), test_size=args.val_fraction,
                                          stratify=cat_train_all, random_state=args.seed)
    train_docs = [training_documents[i] for i in idx_train]
    val_docs = [training_documents[i] for i in idx_val]
    cat_train = cat_train_all[idx_train]
    cat_val = cat_train_all[idx_val]
    _LOGGER.info(f"Documents: {len(train_docs)} for training pairs, {len(val_docs)} for validation pairs, "
                 f"{len(testing_documents)} for test pairs")
    # SPLIT THE DOCUMENTS BEFORE PAIRING
    # Validation and test pairs are built only from documents the model never sees in training,
    # so the model cannot score well by recognising individual training documents.

    vectorizer_X = CountVectorizer(tokenizer=lambda s: s, token_pattern=None, ngram_range=(1, args.max_ngram),
                                   lowercase=False, binary=True, min_df=args.min_df)
    X_train_docs = vectorizer_X.fit_transform(train_docs)
    X_val_docs = vectorizer_X.transform(val_docs)
    X_test_docs = vectorizer_X.transform(testing_documents)
    feature_names = vectorizer_X.get_feature_names_out()
    _LOGGER.info(f"Vocabulary size: {len(feature_names)} n-grams")
    # CREATE THE BINARY WORD FEATURES FOR SINGLE DOCUMENTS
    # Each document becomes a row of 1s and 0s showing which words it contains. The vocabulary
    # is learned from the training documents only.

    n_features = min(args.features, X_train_docs.shape[1])
    SKB = SelectKBest(chi2, k=n_features)
    SKB.fit(X_train_docs, cat_train)
    selected_features = SKB.get_support(indices=True)
    X_train_docs = SKB.transform(X_train_docs).toarray().astype(np.uint8)
    X_val_docs = SKB.transform(X_val_docs).toarray().astype(np.uint8)
    X_test_docs = SKB.transform(X_test_docs).toarray().astype(np.uint8)
    names = pair_feature_names(feature_names, selected_features, args.pair_encoding)
    _LOGGER.info(f"Documents with no selected words: {int((X_train_docs.sum(axis=1) == 0).sum())} train, "
                 f"{int((X_test_docs.sum(axis=1) == 0).sum())} test")
    # CHOOSE THE WORDS
    # Chi-squared keeps the n_features words most associated with the newsgroup categories,
    # using single training documents and their category labels. Words that separate topics
    # are the ones that help decide whether two documents share a topic.
    # Both documents are described with the same lexicon, so their vectors line up word by word,
    # which bundling needs.

    rng = np.random.default_rng(args.seed)
    X_train, Y_train, _ = make_pairs(X_train_docs, cat_train, args.train_pairs, rng, args.pair_encoding)
    X_val, Y_val, val_index = make_pairs(X_val_docs, cat_val, args.val_pairs, rng, args.pair_encoding)
    X_test, Y_test, test_index = make_pairs(X_test_docs, cat_test, args.test_pairs, rng, args.pair_encoding)
    n_columns = X_train.shape[1]
    _LOGGER.info(f"Pairs ({args.pair_encoding}): {len(Y_train)} train, {len(Y_val)} validation, {len(Y_test)} test, "
                 f"{n_columns} features each, {100 * Y_train.mean():.1f}% 'same' in training, "
                 f"{X_train.sum(axis=1).mean():.1f} ones per pair on average")
    # GENERATE THE PAIRS
    # Builds the training, validation and test pairs with make_pairs(). They use a fixed seed,
    # so every run (and every rerun with the same settings) sees the same pairs, and differences
    # between runs come only from the model's own randomness.
    # The average number of 1s per pair is also logged: with bundling, it is the size of the
    # union of the two documents' word sets.

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
    # BASELINE
    # A simple non-learned comparison: compute the TF-IDF cosine similarity of the two documents
    # and answer "same" when it is above a threshold. The threshold is chosen on the validation
    # pairs. This shows how much of the task can be solved just by measuring word overlap.

    clauses = args.number_of_clauses
    structure = ((tm.AND_GROUP, n_columns), (tm.AND_GROUP, 1))
    literals = 2 * n_columns if args.append_negated else n_columns
    estimate = clauses * ((literals - 1) // 32 + 1) * args.number_of_state_bits * 4 * 2
    _LOGGER.info(f"Estimated GPU memory for automata: {estimate / 1024**2:.0f} MB (lower bound)")
    # VANILLA CLAUSE STRUCTURE
    # The same structure as the sample's --vanilla mode: one AND group spanning every column
    # of the pair vector, so each clause is a single AND of literals. The group size is the
    # number of columns in the pair vector: the lexicon size for bundle, twice that for
    # concat. This matches the announcement's (tm.AND_GROUP, features), and the library requires
    # it to equal the number of input columns.
    # The memory estimate covers the automata of both classes ("different" and "same").

    stats_name = (f"newsgroups_pairs_statistics_{args.pair_encoding}_{clauses}_{args.T}_{args.s:.2f}_"
                  f"{args.number_of_state_bits}_{n_features}_{args.train_pairs}.txt")
    f = open(stats_name, "w")
    f.write("run epoch validation_accuracy\n")
    # STATISTICS FILE
    # As in the sample, results are written to a text file named after the main settings,
    # including the pair encoding.
    # Each line holds the run, the epoch and the validation accuracy, and the test result is
    # added at the end of each run.

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
        # CREATE THE MODEL
        # Same call as the sample. With labels 0 and 1, MultiClassTsetlinMachine trains two
        # models, one for "different" and one for "same", and predicts the class whose model
        # gives the higher vote total. np.random.seed fixes the library's sampling for this run.

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
        # TRAIN AND VALIDATE
        # As in the sample, each call to fit() is one epoch. Accuracy is measured on the
        # validation pairs, which replaces the sample's test accuracy during training.

        Y_pred = model.predict(X_test)
        test_acc = 100 * accuracy_score(Y_test, Y_pred)
        test_f1 = 100 * f1_score(Y_test, Y_pred, average="macro")
        test_results.append((test_acc, test_f1))
        _LOGGER.info(f"Run {r + 1}: best validation accuracy {best_val:.2f}% at epoch {best_epoch}")
        _LOGGER.info(f"Run {r + 1}: test accuracy {test_acc:.2f}%, macro-F1 {test_f1:.2f}%")
        f.write(f"# run {r} test_accuracy {test_acc:.2f} test_macro_f1 {test_f1:.2f}\n")
        f.flush()
        # TEST THE RUN
        # The test pairs are scored once, after the run's final epoch. To use the best validation
        # epoch instead, rerun with --epochs set to it.
    f.close()

    if args.runs > 1:
        accs = np.array([a for a, _ in test_results])
        f1s = np.array([b for _, b in test_results])
        print(f"\nTest accuracy over {args.runs} runs: {accs.mean():.2f} ± {accs.std():.2f}")
        print(f"Test macro-F1 over {args.runs} runs: {f1s.mean():.2f} ± {f1s.std():.2f}")
    # SUMMARY ACROSS RUNS
    # Training is random, so the mean and standard deviation over several runs are a more
    # reliable result than any single run.

    print(f"\nLast run - test accuracy: {test_acc:.2f}, macro-F1: {test_f1:.2f}\n")
    print(classification_report(Y_test, Y_pred, target_names=PAIR_CLASSES, digits=3))
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(f"               different   same")
    print(f"  different   {cm[0, 0]:9d}  {cm[0, 1]:5d}")
    print(f"  same        {cm[1, 0]:9d}  {cm[1, 1]:5d}")
    # DETAILED RESULTS FOR THE LAST RUN
    # Precision, recall and F1 for "different" and "same", and a 2 x 2 confusion matrix.
    # For example, many "same" pairs predicted as "different" means the model misses topic matches.

    test_cat_a = cat_test[test_index[:, 0]]
    same_mask = Y_test == 1
    print("\nTest accuracy on 'same' pairs by newsgroup:")
    for c, name in enumerate(category_names):
        m = same_mask & (test_cat_a == c)
        if m.sum():
            print(f"  {name:28s} {100 * (Y_pred[m] == 1).mean():6.2f}%  ({int(m.sum())} pairs)")
    # ACCURACY PER NEWSGROUP
    # Shows which newsgroups the model recognises as matching most easily. Groups with very
    # distinctive words (such as hockey or cryptography) usually score higher than broad or
    # overlapping ones (such as the comp.* groups).

    if args.save_model:
        with open(args.save_model, "wb") as mf:
            pickle.dump({"model": model.save(), "names": names, "category_names": category_names,
                         "pair_encoding": args.pair_encoding}, mf)
        _LOGGER.info(f"Saved the last run's model to {args.save_model}")
    # SAVE THE MODEL (OPTIONAL)
    # Uses the library's save() function and stores the column names and pair encoding alongside
    # it, so the clauses can be explained later without retraining.

    if args.explain:
        print("\nLearned literals and example clauses (last run):")
        for class_id, name in enumerate(PAIR_CLASSES):
            print_class_explanation(model.tms[class_id], name, names, args)
    # EXPLANATIONS
    # model.tms holds the "different" model (0) and the "same" model (1). For each, this prints
    # how much the clauses rely on negated literals, the most important columns in the four
    # polarity groups, and example clauses. The column prefixes follow the encoding (OR:,
    # or A: and B:). For example, with bundling a "same" clause such as ¬OR:hockey ∧ ¬OR:god
    # rules out topics that neither document mentions, which is reasoning by elimination.
    # With concat the same rule needs four literals: ¬A:hockey ∧ ¬B:hockey ∧ ¬A:god ∧ ¬B:god.
