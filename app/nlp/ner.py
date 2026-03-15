from transformers import pipeline

ner = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",
    framework="pt",
    device=-1
)

def extract_entities(text: str):
    entities = ner(text)
    drugs = []
    for ent in entities:
        if ent["entity_group"] in {"DRUG", "CHEMICAL"}:
            drugs.append(ent["word"].lower())
    return list(set(drugs))
