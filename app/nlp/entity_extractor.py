from app.nlp.ner import extract_entities
from app.nlp.brand_matcher import BrandMatcher

class EntityExtractor:
    def __init__(self):
        self.brand_matcher = BrandMatcher()

    def extract(self, text: str):
        ner_entities = extract_entities(text)
        brand_entities = self.brand_matcher.extract(text)

        # Merge + deduplicate
        return list(set(ner_entities + brand_entities))
