# Medical Knowledge Extraction (Dataset-Driven)

This guide shows how to build a dataset-driven medical knowledge structure from the medicines CSV. No hardcoded symptom lists are used; everything is learned from the `uses` column.

## Artifacts Produced

Outputs are written to `indexes/` by default:

- `indexes/conditions.json` (unique condition phrases)
- `indexes/conditions.index` (FAISS index of condition embeddings)
- `indexes/condition_clusters.json` (cluster label -> list of condition phrases)
- `indexes/condition_to_cluster.json` (condition phrase -> cluster label)
- `indexes/treatment_groups.json` (generic name -> brands, uses, warnings)

## Build Command

```powershell
python scripts/build_medical_knowledge.py --csv app/data/medicines_final.csv
```

Common options:

```powershell
python scripts/build_medical_knowledge.py --csv app/data/medicines_final.csv --cluster-method agglomerative --cluster-distance 0.35
python scripts/build_medical_knowledge.py --csv app/data/medicines_final.csv --skip-clusters
python scripts/build_medical_knowledge.py --csv app/data/medicines_final.csv --skip-treatments
```

## Query Pipeline (Runtime)

Use `app/assistant/knowledge_assistant.py` to run a dataset-driven query flow:

```python
from app.assistant.knowledge_assistant import MedicalKnowledgeAssistant

assistant = MedicalKnowledgeAssistant()
response = assistant.run("medicine for cough and fever", language_hint="en")
print(response)
```

The pipeline is:

1. Language detection and translation to English
2. Condition FAISS search
3. Condition cluster selection
4. Query expansion using cluster variants
5. Medicine FAISS search
6. Group results by `generic_name`
7. Attach home care advice (from `data/home_advice.json`)

## Example Output

```json
{
  "query": "medicine for cough and fever",
  "condition_cluster": "cough",
  "possible_conditions": ["dry cough", "cough with mucus", "productive cough"],
  "treatments": [
    {
      "generic_name": "Dextromethorphan",
      "brands": ["Benadryl DM", "Robitussin DM"],
      "uses": "Treatment of dry cough",
      "warnings": "May cause drowsiness"
    }
  ],
  "home_care": ["Drink warm fluids", "Steam inhalation"],
  "disclaimer": "Educational information only. This is not a prescription. Consult a qualified clinician for medical advice."
}
```

## Notes

- The extraction step removes leading phrases like "treatment of" and splits multiple conditions using commas or "and".
- Clusters are labeled automatically using the most frequent condition in the cluster.
- `treatment_groups.json` groups brand names by `generic_name` and attaches common uses and warnings.
