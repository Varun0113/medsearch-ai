# MedSearch AI

MedSearch AI is a multilingual medical information assistant built with FastAPI and a lightweight frontend.

It is designed to provide educational medical information, not prescriptions. The application accepts free-text queries, normalizes multilingual input to English, performs semantic search over the medicines dataset (FAISS + Sentence Transformers), extracts conditions from the `uses` text, groups medicines by `generic_name`, checks urgent warnings, and returns a structured response for the UI.

## Safety Position

- The system does not prescribe medicines.
- Medicines are shown only as commonly used informational references.
- High-risk symptoms trigger an urgent-care override and suppress medicine output.
- The frontend renders UI only and does not embed medical datasets.
- Medical knowledge is loaded from JSON files in `/data`, not hardcoded in source.

## Pipeline

```text
User Query
-> Language Detection
-> Translation to English
-> Semantic Search over medicines dataset
-> Extract condition phrases from `uses`
-> Group medicines by `generic_name`
-> Home care mapping (category advice + fallback)
-> Warning Checks
-> Translation back to the user language
-> Structured JSON response
```

## Features

- Multilingual support for English, Hindi, and Gujarati
- Dataset-driven semantic search (FAISS + Sentence Transformers)
- Possible condition extraction from medicine `uses` text
- Treatment context from dataset `uses` strings
- Common medicine information grouped by `generic_name`
- Home care mapping via `data/homecare.json` with generic fallback
- Urgent symptom override for emergency red flags
- Structured API output for UI rendering

## Project Structure

```text
medical_ai/
|-- app/
|   |-- main.py
|   |-- core/pipeline.py
|   |-- data_loader.py
|   |-- medical/
|   |   |-- knowledge_base.py
|   |   |-- symptom_extractor.py
|   |   |-- condition_predictor.py
|   |   |-- medicine_info.py
|   |   |-- risk_detector.py
|   |   |-- interaction_checker.py
|   |-- nlp/
|   |   |-- language.py
|   |   |-- translator.py
|   |-- schemas/api.py
|-- data/
|   |-- symptoms.json
|   |-- conditions.json
|   |-- treatments.json
|   |-- medicines.json
|   |-- warnings.json
|   |-- homecare.json
|   |-- homecare_mapping.json
|-- frontend/
|   |-- index.html
|   |-- assets/
|   |   |-- app.js
|   |   |-- styles.css
|-- tests/
|   |-- test_pipeline.py
|-- requirements.txt
```

## Datasets And Indexes

Large datasets and FAISS index files are excluded from GitHub to avoid size limits. Download or generate the contents of `data/` and `indexes/` separately before running the full pipeline.

## Installation

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API

### `POST /query`

Request body:

```json
{
  "query": "I have fever and headache for 2 days",
  "language": "en",
  "patient": {
    "age": 32,
    "gender": "female",
    "pregnancy_status": "not_pregnant",
    "allergies": ["penicillin"],
    "existing_conditions": ["stomach ulcer"]
  }
}
```

Note: The `patient` object is accepted for compatibility, but the semantic search pipeline does not use it yet.

Response shape:

```json
{
  "symptoms": [],
  "conditions": [
    {
      "condition": "type 2 diabetes mellitus",
      "confidence": 0.86,
      "reason": "Matched medicine uses such as Treatment of Type 2 Diabetes Mellitus."
    }
  ],
  "treatments": [
    "Treatment of Type 2 Diabetes Mellitus"
  ],
  "otc_medicines": [],
  "prescription_medicines": [
    {
      "name": "Metformin",
      "type": "Prescription",
      "purpose": "Treatment of Type 2 Diabetes Mellitus",
      "safety_notes": [],
      "related_conditions": ["type 2 diabetes mellitus"],
      "patient_specific_warning": ""
    }
  ],
  "home_care": [
    "Consult a healthcare professional for proper diagnosis and treatment."
  ],
  "warnings": [],
  "interaction_warnings": []
}
```

### `POST /semantic-query`

Dataset-driven semantic search endpoint that works for symptoms, diseases, and mixed queries without hardcoded symptom lists.

Request body:

```json
{
  "query": "diabetes",
  "language": "en"
}
```

Response shape:

```json
{
  "query": "diabetes",
  "possible_conditions": [
    "type 2 diabetes",
    "diabetes mellitus"
  ],
  "medicines": [
    {
      "generic_name": "Metformin",
      "brands": ["Glycomet", "Glucophage"],
      "uses": "Treatment of Type 2 Diabetes",
      "warnings": ""
    }
  ],
  "home_care": [
    "Consult a healthcare professional for proper diagnosis and treatment."
  ]
}
```

## Home Care Mapping

Home care advice is stored separately from the medicines dataset in `data/homecare.json`, and a lightweight condition-to-category map is stored in `data/homecare_mapping.json`. The semantic pipeline uses the detected conditions to select a category and return advice, with a generic fallback if no mapping matches.

## Testing

Run the safety regression checks:

```powershell
python -m unittest tests.test_pipeline
```

## Disclaimer

This project provides AI-generated medical information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
