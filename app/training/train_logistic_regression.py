import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT/ "app" / "data" / "reviews.csv"
MODEL_PATH = PROJECT_ROOT / "app" / "model" / "sentiment_model.joblib"

TEXT_COL = "Review Text"
RATING_COL = "Rating"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    df = df[[TEXT_COL, RATING_COL]].dropna()

    # Remove neutral reviews for binary classification
    df = df[df[RATING_COL] != 3]

    # 1,2 => negative = 0 | 4,5 => positive = 1
    df["label"] = df[RATING_COL].apply(lambda x: 1 if x >= 4 else 0)

    X = df[TEXT_COL].astype(str)
    y = df["label"]

    return X, y


def train_model():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            max_features=50000
        )),
        ("classifier", LogisticRegression(
            C=0.3,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
