from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse
import logging
import pickle
from lzma import LZMAFile
import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def save_model(model, path):
	state_dict = model.save()
	with LZMAFile(path, "wb") as f:
		pickle.dump(state_dict, f)
	print("\nModel saved to %s" % path)


def load_model(path):
	with LZMAFile(path, "rb") as f:
		state_dict = pickle.load(f)

	params = state_dict["params"]
	model = MultiClassTsetlinMachine(
		number_of_clauses=params["number_of_clauses"],
		T=params["T"],
		s=params["s"],
		q=params["q"],
		hierarchy_structure=params["hierarchy_structure"],
		boost_true_positive_feedback=params["boost_true_positive_feedback"],
		number_of_state_bits=params["number_of_state_bits"],
		append_negated=params["append_negated"],
		weighted_clauses=state_dict["weighted_clauses"] or False,
	)
	model.load(state_dict)
	print("Model loaded from %s" % path)
	return model

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=2, type=int)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--T", default=128, type=int)
    parser.add_argument("--s", default=21.1, type=float)
    parser.add_argument("--number-of-alternatives", default=64, type=int)
    parser.add_argument("--number-of-elements", default=16, type=int)
    parser.add_argument("--number-of-concepts", default=2, type=int)
    parser.add_argument("--max-ngram", default=2, type=int)
    parser.add_argument("--noise", default=0.0, type=float)
    parser.add_argument("--constant-update-p", action='store_true')
    parser.add_argument('--binary-inference', action='store_true')
    parser.add_argument('--vanilla', action='store_false')
    parser.add_argument('--and-group-normalization', action='store_true')
    parser.add_argument('--no-clipping', action='store_true')
    parser.add_argument("--lemmatize", action='store_false')

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()


features = args.number_of_elements * 2

#fetch and lemmatize dataset
_LOGGER.info("Loading 20 Newsgroups...")
categories = [
    'talk.religion.misc',
    'alt.atheism',
]
REMOVE = ("headers", "footers", "quotes")

train = fetch_20newsgroups(subset="train", categories=categories, remove=REMOVE)
test = fetch_20newsgroups(subset="test", categories=categories, remove=REMOVE)


if args.lemmatize:
    #lemmatize the text
    _LOGGER.info("Lemmatizing the text")
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    documents_train = []
    for doc in nlp.pipe(train.data, batch_size=200):
        text = ""
        for word in doc:
            if not word.is_alpha or len(word) < 2:
                continue
            if not word.is_stop and not len(word) < 2:
                text += " " + word.lemma_.lower()
        documents_train.append(text[1:])
    train.data = documents_train
    documents_test = []
    for doc in nlp.pipe(train.data, batch_size=200):
        text = ""
        for word in doc:
            if not word.is_alpha or len(word) < 2:
                continue
            if not word.is_stop and not len(word) < 2:
                text += " " + word.lemma_.lower()
        documents_test.append(text[1:])
    train.data = documents_test
    _LOGGER.info("Lemmatization complete!")

vectorizer = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, args.max_ngram),
        lowercase=False,
        binary=True,
)
_LOGGER.info("Producing bit representation")
X_train = vectorizer.fit_transform(train.data)
feature_names = vectorizer.get_feature_names_out()
Y_train = train.target.astype(np.uint32)
X_test = vectorizer.transform(test.data)
Y_test = test.target.astype(np.uint32)
_LOGGER.info("Finished producing bit representation!")

_LOGGER.info("Selecting features")
SKB = SelectKBest(chi2, k=features)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train).toarray().astype(np.uint32)
X_test = SKB.transform(X_test).toarray().astype(np.uint32)
_LOGGER.info("Features selected!")

f = open(
    "multi_concept_statistics_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d_%.2f_%d.txt" % (args.number_of_clauses, args.T, args.s,
                                                                              args.number_of_state_bits, args.vanilla,
                                                                              args.and_group_normalization,
                                                                              args.constant_update_p,
                                                                              args.binary_inference,
                                                                              args.number_of_alternatives,
                                                                              args.number_of_elements, args.noise,
                                                                              args.no_clipping), "w")

for r in range(args.runs):
    seed = np.random.randint(10000)
    if args.vanilla:
        _LOGGER.info("Creating Vanilla Tsetlin Machine")
        tsetlin_machine = MultiClassTsetlinMachine(
            args.number_of_clauses * args.number_of_alternatives * args.number_of_concepts,
            args.T,
            args.s,
            binary_inference=args.binary_inference,
            constant_update_p=args.constant_update_p,
            and_group_normalization=args.and_group_normalization,
            seed=seed,
            number_of_state_bits=args.number_of_state_bits,
            boost_true_positive_feedback=0,
            no_clipping=args.no_clipping,
            append_negated=False,
            hierarchy_structure=(
                (tm.AND_GROUP, features),
                (tm.AND_GROUP, 1)
            )
        )
    else:
        _LOGGER.info("Creating hierarchical Tsetlin Machine")
        tsetlin_machine = MultiClassTsetlinMachine(
            args.number_of_clauses,
            args.T,
            args.s,
            binary_inference=args.binary_inference,
            constant_update_p=args.constant_update_p,
            and_group_normalization=args.and_group_normalization,
            seed=seed,
            number_of_state_bits=args.number_of_state_bits,
            boost_true_positive_feedback=0,
            no_clipping=args.no_clipping,
            append_negated=False,

            hierarchy_structure=(
                (tm.AND_GROUP, features),
                (tm.OR_ALTERNATIVES, args.number_of_alternatives),
                (tm.AND_ALTERNATIVES, args.number_of_concepts)
            )

        )

    print("\nAccuracy over %d epochs:\n" % (args.epochs,))

    for e in range(args.epochs):
        start_training = time()
        tsetlin_machine.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result = 100 * (tsetlin_machine.predict(X_test) == Y_test).mean()
        stop_testing = time()

        tsetlin_machine.print_hierarchy()

        print("\n#%d/%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (r + 1, e + 1, result,
                                                                            stop_training - start_training,
                                                                            stop_testing - start_testing))

        f.write("%d %d %.2f\n" % (r, e, result))
        f.flush()
f.close()
