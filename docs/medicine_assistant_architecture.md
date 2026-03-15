# Multilingual Medical AI: Condition-Aware Medicine Search

This doc describes a production-friendly architecture to transform a semantic medicine search system into a safe medical information assistant.

Goals:
- Normalize highly variable `uses` text into canonical symptom/condition groups.
- Expand queries using group variants to improve recall without losing precision.
- Return results grouped by `generic_name` (not brand), with brand examples.
- Attach home-care advice from a separate file (`data/home_advice.json`).
- Avoid prescribing: show "commonly used" medicines, include disclaimers, and encourage clinician review.

## 1) Offline Preprocessing (Build-Time)

### Inputs
- Medicines CSV (250k+ rows): `brand_name`, `generic_name`, `composition`, `uses`, `warnings`, `manufacturer`, `medicine_type`, `price`, `substitutes`, `pack_size`
- Grouping rules: `data/condition_group_rules.json`

### Outputs (Artifacts)
Build a small artifact file that drives runtime detection + expansion:
- `indexes/condition_artifacts.json`
  - `groups[group].variants[]`: extracted condition phrases from `uses`
  - `groups[group].top_generics[]`: most frequent `generic_name` values linked to that group

Command:
```powershell
python scripts/build_condition_artifacts.py --csv app/data/medicines_final.csv --out indexes/condition_artifacts.json
```

Implementation:
- `app/assistant/condition_groups.py` parses/splits/normalizes `uses` phrases.
- `app/assistant/artifacts.py` builds aggregated artifacts in chunks.

## 2) Runtime Pipeline (FastAPI)

### High-level flow
1. Detect language and translate query to English (keep index + artifacts English).
2. Detect condition groups from the query.
3. Expand query terms using variants for detected groups.
4. Retrieve candidates via FAISS (`app/search/semantic_search.py`).
5. Group results by `generic_name`.
6. Attach home advice (per detected group) from `data/home_advice.json`.
7. Return a structured response and a medical disclaimer.

### Why this works
- The artifact-backed group lexicon turns "cough" into many dataset-specific phrases (dry cough, cough with mucus, allergic cough), improving recall.
- Group-aware filtering and grouping prevents duplicate generics and makes results easier to review.

## 3) Core Runtime Helpers

Load artifacts + rules:
```python
from pathlib import Path
from app.assistant.artifacts import ConditionArtifacts
from app.assistant.condition_groups import ConditionGroupNormalizer

artifacts = ConditionArtifacts.load(Path("indexes/condition_artifacts.json"))
normalizer = ConditionGroupNormalizer(Path("data/condition_group_rules.json"))
```

Detect groups + expand query:
```python
from app.assistant.query_tools import detect_condition_groups, expand_query

groups = detect_condition_groups(query_en, artifacts, normalizer)
expanded = expand_query(query_en, groups, artifacts)
```

Group results by `generic_name`:
```python
from app.assistant.query_tools import group_results_by_generic

grouped = group_results_by_generic(semantic_results)
```

## 4) Suggested API Response JSON

```json
{
  "language": "hi",
  "query_en": "medicine for cough and fever",
  "detected_conditions": [
    {"group": "cough", "evidence": ["cough"]},
    {"group": "fever", "evidence": ["fever"]}
  ],
  "expanded_query_en": "medicine for cough and fever cough dry cough ... fever viral fever ...",
  "home_care": {
    "cough": ["Drink warm fluids.", "Steam inhalation may help relieve congestion."],
    "fever": ["Rest and drink fluids.", "Monitor temperature and overall weakness."]
  },
  "medicines": [
    {
      "generic_name": "Dextromethorphan",
      "category": "allopathy",
      "uses": "Treatment of dry cough",
      "composition": "Dextromethorphan ...",
      "warnings": "Drowsiness ...",
      "best_relevance": 92,
      "brands": [{"name": "Benadryl DM ...", "form": "Syrup", "strength": "", "confidence": "high"}]
    }
  ],
  "disclaimer": "Educational information only. Not a prescription. Consult a qualified clinician."
}
```

## 5) Safety Notes (Recommended)
- Never return dose/frequency.
- Always include an explicit disclaimer.
- If the query includes emergency red flags (e.g., chest pain, trouble breathing), override with urgent-care guidance and suppress medicine results.
- Track "why" a group was detected (`evidence`) to keep behavior debuggable.

## 6) Production Hardening Checklist
- Store metadata as Parquet/Arrow instead of pickle for faster loads.
- Keep embedding model + translator loaded once at startup.
- Cache translations for repeated short queries.
- Add unit tests for:
  - `uses` parsing and splitting
  - normalization rules
  - query expansion output
  - grouping by `generic_name`

