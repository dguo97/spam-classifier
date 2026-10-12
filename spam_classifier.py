import argparse
import io
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_PATH = Path(__file__).with_name("spam_model.pkl")


def make_pipeline():
    return Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42)),
    ])


def train_model():
    """Download the dataset, evaluate, train a model, and return it."""
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), sep="\t", header=None, names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Cross-validation on the full dataset: a steadier estimate than a single split.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        make_pipeline(), df["text"], df["label"], cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
    )
    print("5-fold cross-validation (mean +/- std; precision, recall and F1 are for spam):")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = scores[f"test_{metric}"]
        print(f"  {metric:<10} {values.mean():.4f} +/- {values.std():.4f}")

    # Hold-out split: train the model, then report detailed results on unseen messages.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = make_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nHold-out test set (20% of messages):")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual ham", "actual spam"],
        columns=["predicted ham", "predicted spam"],
    ))
    return model


def classify(model, message):
    """Return the predicted label and the model's confidence in it."""
    ham_prob, spam_prob = model.predict_proba([message])[0]
    if spam_prob >= 0.5:
        return "SPAM", spam_prob
    return "HAM", ham_prob


def main():
    parser = argparse.ArgumentParser(description="SMS spam classifier")
    parser.add_argument(
        "--retrain", action="store_true", help="ignore the saved model and train a new one"
    )
    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.retrain:
        print(f"Loading saved model from {MODEL_PATH.name} (use --retrain to rebuild it)")
        model = joblib.load(MODEL_PATH)
    else:
        model = train_model()
        joblib.dump(model, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH.name}")

    print("\nEnter a message to check (blank line or 'quit' to exit).")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            break
        if not message or message.lower() == "quit":
            break
        label, confidence = classify(model, message)
        print(f"{label} ({confidence:.0%})")


if __name__ == "__main__":
    main()
