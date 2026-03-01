"""Inference module for ticket category prediction.

Provides classify_ticket(text, image=None), where image is accepted for
interface compatibility but currently ignored.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Default local model path. Keep everything offline/internal.
DEFAULT_MODEL_PATH = Path("models/ticket_classifier.pkl")


class TicketClassifier:
    """Wrapper around a trained scikit-learn text classification pipeline."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Train the model first using train_model.py"
            )
        with self.model_path.open("rb") as f:
            self.pipeline = pickle.load(f)

    def classify_ticket(self, text: str, image: str | None = None) -> dict[str, Any]:
        """Predict ticket category and confidence.

        Args:
            text: Ticket description text.
            image: Optional base64 image payload in HTML body.
                   This is intentionally ignored for now.

        Returns:
            {
                "classification": "<predicted_category>",
                "confidence": <percentage_float>
            }
        """
        # `image` is currently unused by design. Retained for future multimodal support.
        _ = image

        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        predicted_label = self.pipeline.predict([text])[0]

        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        predicted_index = list(classes).index(predicted_label)
        confidence_pct = round(float(probabilities[predicted_index]) * 100, 2)

        return {
            "classification": str(predicted_label),
            "confidence": confidence_pct,
        }


_default_classifier: TicketClassifier | None = None


def _get_default_classifier() -> TicketClassifier:
    """Lazy-load classifier on first inference call."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = TicketClassifier()
    return _default_classifier


def classify_ticket(text: str, image: str | None = None) -> dict[str, Any]:
    """Public inference function requested by the API contract."""
    return _get_default_classifier().classify_ticket(text=text, image=image)


if __name__ == "__main__":
    sample_text = "I cannot log in after password reset and keep seeing access denied"
    result = classify_ticket(sample_text)
    print("Input:", sample_text)
    print("Output:", result)
