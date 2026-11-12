import argparse
import csv
import logging
import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
# IMPORTS
# - scikit-learn supplies the 20 Newsgroups dataset and the vectorizer that records which
#   words each post contains.
# - csv writes the full table of results.
# - spaCy is imported later, inside preprocess(), and only when posts need to be processed.

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# LOGGING
# Creates a logger for this file and shows messages at INFO level and above.

REMOVE = ("headers", "footers", "quotes")
# METADATA TO STRIP
# The same setting as the Tsetlin Machine scripts, so the words counted here are the words
# the model sees. Without it, names and email addresses would appear as "frequent words".


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
# TOKENIZE (AND OPTIONALLY LEMMATIZE) THE POSTS
# Identical to the Tsetlin Machine scripts, so both work on exactly the same words.


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
# CACHE THE PROCESSED POSTS
# Uses the same cache files and naming as the Tsetlin Machine scripts. Run this script from
# the same folder (or pass the same --cache-dir) and it loads the posts you already processed.


def bool_flag(parser, name, default, help_text):
    parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)
# TRUE/FALSE SETTINGS
# Creates a pair of flags such as --lemmatize and --no-lemmatize.


if __name__ == "__main__":
    # ENTRY POINT
    # The code below runs only when this file is executed directly, not when imported.

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="train", choices=["train", "test", "all"],
                        help="which posts to count (default: training posts only)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="newsgroups to use, e.g. sci.crypt sci.med rec.autos (default: all 20)")
    parser.add_argument("--min-df", default=5, type=int, help="ignore words in fewer posts than this")
    parser.add_argument("--top", default=15, type=int, help="literals to print per list and category")
    parser.add_argument("--absence-threshold", default=0.95, type=float,
                        help="minimum share of a category's posts that must lack the word")
    parser.add_argument("--output", default="classwise_literals.csv", type=str)
    bool_flag(parser, "lemmatize", True, "reduce words to their lemma with spaCy")
    bool_flag(parser, "keep-stopwords", False, "keep words such as 'the' and 'not'")
    parser.add_argument("--cache-dir", default=".cache_20newsgroups", type=str)
    args = parser.parse_args()
    # COMMAND-LINE SETTINGS
    # - subset: count training posts by default, so the analysis does not look at the test data.
    # - categories: limit the analysis to some newsgroups. "Other categories" then means only
    #   the other chosen ones.
    # - min-df: skips very rare words, whose percentages would be unreliable.
    # - top: how many literals to print in each list.
    # - absence-threshold: for absent-word literals, how reliably the word must be missing from
    #   the category (0.95 = missing from at least 95% of its posts). See the absence section below.
    # - output: CSV file with the statistics for every word and category.
    # - lemmatize, keep-stopwords, cache-dir: must match the Tsetlin Machine script's settings so
    #   the words are the same and the cache is reused.

    subsets = ["train", "test"] if args.subset == "all" else [args.subset]
    documents, labels = [], []
    for subset in subsets:
        data = fetch_20newsgroups(subset=subset, remove=REMOVE, categories=args.categories)
        documents += load_tokens(subset, data.data, args)
        labels.append(data.target)
        class_names = data.target_names
    labels = np.concatenate(labels)
    _LOGGER.info(f"Counting {len(documents)} posts in {len(class_names)} categories")
    # LOAD THE POSTS
    # Loads the chosen posts and their category labels, using the cached tokens where possible.
    # With --subset all, the training and test posts are combined.

    vectorizer = CountVectorizer(tokenizer=lambda s: s, token_pattern=None, lowercase=False,
                                 binary=True, min_df=args.min_df)
    X = vectorizer.fit_transform(documents).tocsr()
    words = vectorizer.get_feature_names_out()
    _LOGGER.info(f"Vocabulary: {len(words)} words appearing in at least {args.min_df} posts")
    # RECORD WHICH WORDS EACH POST CONTAINS
    # Each post becomes a row of 1s and 0s (present or absent), the same binary form the
    # Tsetlin Machine sees. Counting a word once per post, rather than every time it appears,
    # means a frequency below is "the share of posts that contain the word".

    n_classes = len(class_names)
    docs_per_class = np.bincount(labels, minlength=n_classes)
    counts = np.vstack([np.asarray(X[labels == c].sum(axis=0)).ravel() for c in range(n_classes)])
    total = counts.sum(axis=0)
    p_in = counts / docs_per_class[:, None]
    p_out = (total[None, :] - counts) / (len(labels) - docs_per_class)[:, None]
    # WORD FREQUENCY PER CATEGORY
    # For every category c and word w:
    # - p_in[c, w]  = share of c's posts that contain w, P(w | c).
    # - p_out[c, w] = share of all other categories' posts that contain w, P(w | not c).
    # The absent-word literal ¬w has the opposite frequencies: P(¬w | c) = 1 - p_in.

    rows = []
    for c, name in enumerate(class_names):
        print(f"\n=== {name} ({docs_per_class[c]} posts) ===")

        order = np.argsort(-p_in[c])[:args.top]
        print("  Most frequent words (share of posts in this category | in other categories):")
        print("   " + ", ".join(f"{words[w]} ({100 * p_in[c, w]:.0f}% | {100 * p_out[c, w]:.0f}%)" for w in order))
        # MOST FREQUENT WORDS
        # The words that appear in the largest share of this category's posts. Some are frequent
        # everywhere (for example "know" or "think"), so compare the two percentages: a word
        # that is equally common in other categories is frequent, but not typical of this one.

        presence_score = p_in[c] - p_out[c]
        order = np.argsort(-presence_score)[:args.top]
        print("  Most typical words (frequent here, rare elsewhere):")
        print("   " + ", ".join(f"{words[w]} ({100 * p_in[c, w]:.0f}% | {100 * p_out[c, w]:.0f}%)" for w in order))
        # MOST TYPICAL WORDS (PRESENCE LITERALS)
        # Ranked by how much more often the word appears here than in other categories:
        # P(w | c) - P(w | not c). These are the plain-word literals most likely to support this
        # category, such as "hockey" for rec.sport.hockey.

        absence_score = p_out[c] - p_in[c]
        reliable = (1 - p_in[c]) >= args.absence_threshold
        candidates = np.where(reliable)[0]
        order = candidates[np.argsort(-absence_score[candidates])][:args.top]
        print(f"  Most typical absences (missing from at least {100 * args.absence_threshold:.0f}% of posts here, "
              f"common elsewhere):")
        print("   " + ", ".join(f"¬{words[w]} ({100 * (1 - p_in[c, w]):.0f}% | {100 * (1 - p_out[c, w]):.0f}%)"
                                for w in order))
        # MOST TYPICAL ABSENCES (NEGATED LITERALS)
        # Almost every word is absent from almost every post, so raw frequency cannot rank ¬w:
        # nearly all would score close to 100%. Instead:
        # - Keep only words missing from at least absence-threshold of this category's posts.
        #   With s = 1, a literal is forgotten whenever it is false on the clause's own class, so
        #   only literals that are almost always true there can survive.
        # - Rank them by how much more often the word appears in other categories:
        #   P(w | not c) - P(w | c). These ¬w literals rule out the most posts from other
        #   categories, which is what reasoning by elimination relies on.
        # The percentages shown are P(¬w | c) and P(¬w | not c).

        for w in range(len(words)):
            rows.append((name, words[w], counts[c, w], f"{p_in[c, w]:.4f}", f"{p_out[c, w]:.4f}",
                         f"{presence_score[w]:.4f}", f"{absence_score[w]:.4f}"))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "word", "posts_with_word", "p_in_category", "p_in_other_categories",
                         "presence_score", "absence_score"])
        writer.writerows(rows)
    _LOGGER.info(f"Wrote statistics for {len(words)} words x {n_classes} categories to {args.output}")
    # SAVE THE FULL TABLE
    # Writes one row per category and word, so you can sort, filter or plot the results in a
    # spreadsheet or with pandas:
    # - presence_score = P(w | c) - P(w | not c): high means w is typical of the category.
    # - absence_score  = P(w | not c) - P(w | c): high means ¬w is typical of the category.
    # The two scores are opposites, so one column is enough to sort either way.
