import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.config import LANG_MAP
from app.nlp.language import detect_language
from app.nlp.translator import Translator
from app.search.semantic_search import SemanticSearchEngine
from app.assistant.query_tools import group_results_by_generic
from app.assistant.uses_extraction import extract_conditions_from_uses
from app.assistant.homecare import load_homecare_categories, load_condition_category_map, get_homecare_advice


@dataclass(frozen=True)
class SemanticMedicineGroup:
    generic_name: str
    brands: List[str]
    uses: str
    warnings: str


class SemanticSearchPipeline:
    def __init__(self) -> None:
        self.translator = Translator()
        self.search_engine = SemanticSearchEngine()
        base_dir = Path(__file__).resolve().parents[2]
        homecare_path = base_dir / "data" / "homecare.json"
        mapping_path = base_dir / "data" / "homecare_mapping.json"
        self.homecare_categories = load_homecare_categories(homecare_path) if homecare_path.exists() else {}
        self.condition_category_map = load_condition_category_map(mapping_path) if mapping_path.exists() else {}
        self.generic_home_care = "Consult a healthcare professional for proper diagnosis and treatment."

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            key = re.sub(r"\s+", " ", str(item).strip().lower())
            if key and key not in seen:
                seen.add(key)
                output.append(str(item).strip())
        return output

    def _resolve_language(self, user_text: str, language_hint: str | None) -> tuple[str, str]:
        hint = (language_hint or "").strip().lower()
        detected_lang = detect_language(user_text)
        if hint in {"hi", "gu"}:
            user_lang = hint
        elif hint == "en" and detected_lang in {"hi", "gu"}:
            user_lang = detected_lang
        else:
            user_lang = hint if hint in LANG_MAP else detected_lang
        return user_lang, LANG_MAP.get(user_lang, "eng_Latn")

    def _to_english(self, text: str, user_lang: str, nllb_lang: str) -> str:
        if user_lang == "en":
            return text[:400]
        try:
            return self.translator.to_english(text[:400], nllb_lang)
        except Exception:
            return text[:400]

    def _extract_conditions_from_results(self, results: list[dict[str, object]], limit: int = 20) -> list[str]:
        conditions: list[str] = []
        for item in results:
            uses_text = str(item.get("description") or "").strip()
            if not uses_text:
                continue
            conditions.extend(extract_conditions_from_uses(uses_text))
            if len(conditions) >= limit * 2:
                break
        return self._dedupe(conditions)[:limit]

    def _home_care_for(self, possible_conditions: list[str]) -> list[str]:
        return get_homecare_advice(
            conditions=possible_conditions,
            category_map=self.condition_category_map,
            categories=self.homecare_categories,
            fallback=self.generic_home_care,
        )

    @staticmethod
    def _grouped_to_response(grouped: list[dict[str, object]]) -> list[SemanticMedicineGroup]:
        response: list[SemanticMedicineGroup] = []
        for entry in grouped:
            response.append(
                SemanticMedicineGroup(
                    generic_name=str(entry.get("generic_name") or "").strip(),
                    brands=[
                        str(item.get("name") or "").strip()
                        for item in (entry.get("brands") or [])
                        if str(item.get("name") or "").strip()
                    ],
                    uses=str(entry.get("uses") or "").strip(),
                    warnings=str(entry.get("warnings") or "").strip(),
                )
            )
        return response

    def run(self, query: str, language_hint: str | None = None) -> dict[str, object]:
        query = (query or "").strip()
        if not query:
            return {
                "query": "",
                "possible_conditions": [],
                "medicines": [],
            }

        user_lang, nllb_lang = self._resolve_language(query, language_hint)
        query_en = self._to_english(query, user_lang, nllb_lang)

        results = self.search_engine.search(query_en, top_k=40)
        possible_conditions = self._extract_conditions_from_results(results)

        grouped = group_results_by_generic(results)
        medicines = self._grouped_to_response(grouped)

        return {
            "query": query,
            "possible_conditions": possible_conditions,
            "medicines": [medicine.__dict__ for medicine in medicines],
            "home_care": self._home_care_for(possible_conditions),
        }
