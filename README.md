# Internal Ticket Classification (Offline)

This repository contains a fully local/offline Python module for classifying support tickets into these categories:

- Access Issue
- Admin Activities
- Data Fix
- Technical Issue
- Integration Issues
- User Guide
- Process Exception
- Reporting
- Support

## 1) Training Script

`ticket_classifier/train_model.py` trains a text classifier from a CSV with columns:

- `description`
- `category`

### Run training

```bash
python -m ticket_classifier.train_model \
  --csv-path ./training_data.csv \
  --model-path ./models/ticket_classifier.pkl
```

This creates a serialized model pipeline on disk (`pickle` file).

---

## 2) Inference Module

`ticket_classifier/inference.py` exposes:

```python
classify_ticket(text, image=None)
```

- Accepts ticket text and optional image (base64 in HTML body)
- Ignores image for now
- Returns:

```python
{
  "classification": "<predicted_category>",
  "confidence": <percentage>
}
```

### Example usage

```python
from ticket_classifier import classify_ticket

ticket_text = "User cannot access dashboard after role change"
result = classify_ticket(ticket_text, image="<base64-image-string>")
print(result)
```

### Example output

```python
{
  "classification": "Access Issue",
  "confidence": 87.34
}
```

---

## 3) Model/Tech Stack

- **Language**: Python
- **ML library**: scikit-learn
- **Vectorizer**: TF-IDF (`TfidfVectorizer`)
- **Classifier**: Logistic Regression (`LogisticRegression`)

---

## 4) Brief Architecture

1. **Data ingestion**: CSV is loaded with Python's built-in `csv` module and validated for required columns.
2. **Preprocessing + features**: Text is transformed into TF-IDF vectors (unigrams+bigrams).
3. **Learning**: Logistic Regression is trained on historical labeled tickets.
4. **Persistence**: The full pipeline (vectorizer + classifier) is saved locally with `pickle`.
5. **Inference**: `classify_ticket` loads model, predicts label + probability, and returns category/confidence.

All processing runs **internally and offline**, with no cloud/API dependency.

---

## 5) Bi-Weekly Operations Report Web Page

A ready-to-use static report page is available at:

- `biweekly_report.html`

### Features

- Matches the same presentation structure as your existing slide.
- Editable fields for period, owner, incident counts, risks, and activities.
- Saves report history in browser localStorage.
- Keeps the latest **5 reports** (covers approximately **last 10 weeks** of bi-weekly updates).
- Load/delete previous reports and export current report as JSON.

### Open in browser

```bash
xdg-open biweekly_report.html
```

Or just double-click the file in your file explorer.
