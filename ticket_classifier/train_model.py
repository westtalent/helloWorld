"""Training script for the internal ticket classification model.

This script:
1. Reads a CSV file containing `description` and `category` columns.
2. Trains a scikit-learn pipeline (TF-IDF + Logistic Regression).
3. Saves the trained pipeline to disk for offline inference.

Usage example:
    python -m ticket_classifier.train_model \
        --csv-path ./training_data.csv \
        --model-path ./models/ticket_classifier.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_training_pipeline() -> Pipeline:
    """Create the text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def load_training_data(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load and validate training data from CSV using stdlib csv module."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    texts: list[str] = []
    labels: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"description", "category"}
        field_names = set(reader.fieldnames or [])
        missing = required_columns.difference(field_names)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in reader:
            description = (row.get("description") or "").strip()
            category = (row.get("category") or "").strip()

            if description and category:
                texts.append(description)
                labels.append(category)

    if not texts:
        raise ValueError("Training dataset is empty after removing invalid rows.")

    return texts, labels


def train_and_save_model(csv_path: Path, model_path: Path) -> None:
    """Train the classifier and save the fitted pipeline to disk."""
    texts, labels = load_training_data(csv_path)

    pipeline = build_training_pipeline()
    pipeline.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model trained on {len(texts)} tickets and saved to: {model_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ticket classification model.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to CSV containing description,category columns.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/ticket_classifier.pkl"),
        help="Output path for the trained model file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save_model(args.csv_path, args.model_path)
