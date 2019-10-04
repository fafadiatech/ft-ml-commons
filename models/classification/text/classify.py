import argparse
import pandas as pd
import numpy as np

from pre_process import load_dataset
from feature_gen import gen_counting_features

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument("train", help="training dataset in TSV format {label, text}")
parser.add_argument("test", help="training dataset in TSV format {label, text}")
args = parser.parse_args()


def evaluate(model_context, X_train, y_train, X_test, y_test, detailed_report=False):
    """
    this function is used to generate 
    """
    print(f"Evaluating {model_context['name']}:")
    clf = model_context["model"]

    clf.fit(X_train, y_train)

    y_test_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_predicted)
    print(f"Accuracy Score:{accuracy}")

    if detailed_report:
        print(confusion_matrix(y_test, y_test_predicted))

if __name__ == "__main__":

    models = [
        {
        "model": BernoulliNB(),
        "name": "Bernoulli Naive Bayes Classifer"
        },
        {
        "model": SGDClassifier(max_iter=100, tol=1e-3),
        "name": "SGD Classifer"
        },
        {
        "model": RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42),
        "name": "RF Classifer"
        },
    ]

    X_train, y_train = load_dataset(args.train)
    X_test, y_test = load_dataset(args.test)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

    X_train = vectorizer.fit_transform(X_train.values.astype('U'))
    X_test = vectorizer.transform(X_test.values.astype('U'))

    for model_context in models:
        evaluate(model_context, X_train, y_train, X_test, y_test)