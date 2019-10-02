import argparse
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument("train", help="training dataset in TSV format {label, text}")
parser.add_argument("test", help="training dataset in TSV format {label, text}")
args = parser.parse_args()

def load_dataset(file_to_read):
    """
    this method is used to load dataset into pandas frame
    """
    df = pd.read_csv(file_to_read, sep="\t")
    return df.iloc[:, 1], df.iloc[:, 0]

def evaluate(model_context, X_train, y_train, X_test, y_test):
    """
    this function is used to generate 
    """
    print(f"Evaluating {model_context['name']}:")
    vectorizer = TfidfVectorizer(stop_words='english', min_df=30)
    svd = TruncatedSVD(n_components=400, n_iter=7, random_state=42)
    X_train = svd.fit_transform(vectorizer.fit_transform(X_train))
    clf = model_context["model"]
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(scores)


if __name__ == "__main__":

    models = [
        {
        "model": RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42),
        "name": "RF Classifer"
        },
        {
        "model": BernoulliNB(),
        "name": "Bernoulli Naive Bayes Classifer"
        },
        {
        "model": SVC(gamma='scale', probability=True),
        "name": "Support Vector Classifer"
        },
        {
        "model": SGDClassifier(max_iter=100, tol=1e-3),
        "name": "SGD Classifer"
        },
    ]

    X_train, y_train = load_dataset(args.train)
    X_test, y_test = load_dataset(args.test)

    for model_context in models:
        evaluate(model_context, X_train, y_train, X_test, y_test)