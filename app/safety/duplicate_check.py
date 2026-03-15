import re
from collections import defaultdict


class DuplicateMedicineChecker:
    @staticmethod
    def _clean_component(component: str) -> str:
        component = re.sub(r"\([^)]*\)", "", component or "")
        component = re.sub(r"[^a-zA-Z0-9 ]+", " ", component)
        return re.sub(r"\s+", " ", component).strip().lower()

    @classmethod
    def normalize_generic(cls, generic_name: str):
        parts = re.split(r"\s*\+\s*", generic_name or "")
        cleaned = [cls._clean_component(part) for part in parts]
        return [part for part in cleaned if part]

    def check(self, results):
        ingredient_to_brands = defaultdict(set)

        for result in results:
            generic = result.get("generic") or result.get("generic_name")
            if not generic:
                continue

            ingredients = self.normalize_generic(str(generic))
            if not ingredients:
                continue

            brand_names = {
                b.get("name", "").strip()
                for b in result.get("brands", [])
                if b.get("name")
            }
            if not brand_names:
                continue

            for ingredient in ingredients:
                ingredient_to_brands[ingredient].update(brand_names)

        return {
            ingredient: sorted(brands)
            for ingredient, brands in ingredient_to_brands.items()
            if len(brands) > 1
        }
