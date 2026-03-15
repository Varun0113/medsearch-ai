import re
from collections import OrderedDict, Counter
from copy import deepcopy
from pathlib import Path

from app.config import LANG_MAP
from app.data_loader import MedicalDataLoader, get_data_loader
from app.nlp.language import detect_language
from app.nlp.translator import Translator
from app.search.semantic_search import SemanticSearchEngine
from app.assistant.query_tools import group_results_by_generic
from app.assistant.homecare import load_homecare_categories, load_condition_category_map, get_homecare_advice
from app.assistant.uses_extraction import extract_conditions_from_uses


class MedSearchPipeline:
    _GENERIC_HOME_CARE = "Consult a healthcare professional for proper diagnosis and treatment."
    def __init__(self, data_loader: MedicalDataLoader | None = None) -> None:
        self.translator = Translator()
        self.data_loader = data_loader or get_data_loader()
        self.data_loader.load_datasets()
        self.semantic_search = SemanticSearchEngine()
        base_dir = Path(__file__).resolve().parents[2]
        homecare_path = base_dir / "data" / "homecare.json"
        mapping_path = base_dir / "data" / "homecare_mapping.json"
        self.homecare_categories = load_homecare_categories(homecare_path) if homecare_path.exists() else {}
        self.condition_category_map = load_condition_category_map(mapping_path) if mapping_path.exists() else {}
        self._cache: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._cache_size = 128

    def _cache_get(self, key: str):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return deepcopy(self._cache[key])

    def _cache_set(self, key: str, value: dict[str, object]) -> None:
        self._cache[key] = deepcopy(value)
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            key = re.sub(r"\s+", " ", str(item).strip().lower())
            if key and key not in seen:
                seen.add(key)
                deduped.append(str(item).strip())
        return deduped

    @staticmethod
    def _split_terms(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            parts = re.split(r",|;|\n", value)
            return [part.strip() for part in parts if part.strip()]
        return [str(value).strip()]

    def _normalize_patient(self, patient: dict[str, object] | None) -> dict[str, object]:
        patient = patient or {}
        age = patient.get("age")
        try:
            normalized_age = int(age) if age is not None and str(age).strip() else None
        except (TypeError, ValueError):
            normalized_age = None

        return {
            "age": normalized_age,
            "gender": str(patient.get("gender") or "").strip(),
            "pregnancy_status": str(patient.get("pregnancy_status") or "").strip().lower(),
            "allergies": self._split_terms(patient.get("allergies")),
            "existing_conditions": self._split_terms(patient.get("existing_conditions")),
            "current_medications": self._split_terms(patient.get("current_medications")),
        }

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

    def _translate_text(self, text: str, target_lang: str, memo: dict[str, str]) -> str:
        source = (text or "").strip()
        if not source or target_lang in {"en", "eng_Latn"}:
            return source
        if source in memo:
            return memo[source]
        try:
            translated = self.translator.from_english(source, target_lang).strip()
        except Exception:
            translated = source
        memo[source] = translated or source
        return memo[source]

    def _localize_payload(self, payload: dict[str, object], user_lang: str, target_lang: str) -> dict[str, object]:
        if user_lang == "en":
            return payload

        localized = deepcopy(payload)
        memo: dict[str, str] = {}
        urgent_banner = self.data_loader.get_system_message("urgent_banner")

        localized["symptoms"] = [
            self._translate_text(item, target_lang, memo)
            for item in localized.get("symptoms", [])
        ]
        localized["treatments"] = [
            self._translate_text(item, target_lang, memo)
            for item in localized.get("treatments", [])
        ]
        localized["home_care"] = [
            self._translate_text(item, target_lang, memo)
            for item in localized.get("home_care", [])
        ]
        localized["warnings"] = [
            item if item == urgent_banner else self._translate_text(item, target_lang, memo)
            for item in localized.get("warnings", [])
        ]

        for condition in localized.get("conditions", []):
            condition["condition"] = self._translate_text(condition.get("condition", ""), target_lang, memo)
            condition["reason"] = self._translate_text(condition.get("reason", ""), target_lang, memo)

        for collection in ("otc_medicines", "prescription_medicines"):
            for medicine in localized.get(collection, []):
                medicine["purpose"] = self._translate_text(medicine.get("purpose", ""), target_lang, memo)
                medicine["safety_notes"] = [
                    self._translate_text(note, target_lang, memo)
                    for note in medicine.get("safety_notes", [])
                ]
                medicine["related_conditions"] = [
                    self._translate_text(item, target_lang, memo)
                    for item in medicine.get("related_conditions", [])
                ]
                medicine["patient_specific_warning"] = self._translate_text(
                    medicine.get("patient_specific_warning", ""),
                    target_lang,
                    memo,
                )

        for warning in localized.get("interaction_warnings", []):
            warning["reason"] = self._translate_text(warning.get("reason", ""), target_lang, memo)

        return localized

    @staticmethod
    def _public_medicine_entries(medicines: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "name": item["name"],
                "type": item["type"],
                "purpose": item["purpose"],
                "safety_notes": item["safety_notes"],
                "related_conditions": item["related_conditions"],
                "patient_specific_warning": item["patient_specific_warning"],
            }
            for item in medicines
        ]

    @staticmethod
    def _medicine_type_from_category(category: str) -> str:
        text = str(category or "").lower()
        if "otc" in text or "over the counter" in text or "non prescription" in text:
            return "OTC"
        if "prescription" in text:
            return "Prescription"
        return "Prescription"

    def _semantic_conditions(
        self,
        results: list[dict[str, object]],
        limit: int = 6,
    ) -> tuple[list[dict[str, object]], list[str]]:
        counts: Counter[str] = Counter()
        examples: dict[str, str] = {}

        for item in results:
            uses_text = str(item.get("description") or "").strip()
            if not uses_text:
                continue
            for condition in extract_conditions_from_uses(uses_text):
                counts[condition] += 1
                if condition not in examples:
                    examples[condition] = uses_text

        if not counts:
            return [], []

        max_count = max(counts.values()) or 1
        condition_entries: list[dict[str, object]] = []
        ordered_conditions: list[str] = []

        for condition, count in counts.most_common(limit):
            confidence = round(min(0.95, 0.5 + 0.45 * (count / max_count)), 2)
            example = examples.get(condition, "")
            reason = (
                f"Matched medicine uses such as {example}."
                if example
                else "Matched medicine uses text in the dataset."
            )
            condition_entries.append(
                {
                    "condition": condition,
                    "confidence": confidence,
                    "reason": reason,
                }
            )
            ordered_conditions.append(condition)

        return condition_entries, ordered_conditions

    def _semantic_treatments(
        self,
        results: list[dict[str, object]],
        limit: int = 8,
    ) -> list[str]:
        uses_list: list[str] = []
        for item in results:
            uses_text = str(item.get("description") or "").strip()
            if uses_text:
                uses_list.append(uses_text)
        return self._dedupe(uses_list)[:limit]

    def _semantic_medicine_entries(
        self,
        grouped_results: list[dict[str, object]],
        related_conditions: list[str],
        limit: int = 12,
    ) -> list[dict[str, object]]:
        medicines: list[dict[str, object]] = []
        for entry in grouped_results[:limit]:
            warnings = str(entry.get("warnings") or "").strip()
            safety_notes = [warnings] if warnings else []
            medicines.append(
                {
                    "name": str(entry.get("generic_name") or "").strip(),
                    "type": self._medicine_type_from_category(str(entry.get("category") or "")),
                    "purpose": str(entry.get("uses") or "Commonly used for symptom relief.").strip(),
                    "safety_notes": safety_notes,
                    "related_conditions": [item for item in related_conditions[:3]],
                    "patient_specific_warning": "",
                }
            )
        return medicines

    def run(
        self,
        user_text: str,
        language_hint: str | None = None,
        patient: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if not user_text or not user_text.strip():
            return {
                "symptoms": [],
                "conditions": [],
                "treatments": [],
                "otc_medicines": [],
                "prescription_medicines": [],
                "home_care": [],
                "warnings": [],
                "interaction_warnings": [],
            }

        normalized_patient = self._normalize_patient(patient)
        patient_key = repr(sorted(normalized_patient.items()))
        cache_key = f"{language_hint}|{patient_key}|{user_text.strip().lower()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        user_lang, nllb_lang = self._resolve_language(user_text, language_hint)
        english_text = self._to_english(user_text, user_lang, nllb_lang)

        semantic_results = self.semantic_search.search(english_text, top_k=40)
        urgent_matches = self.data_loader.detect_urgent_risks(english_text)

        symptoms: list[str] = []
        conditions: list[dict[str, object]] = []
        treatments: list[str] = []
        home_care: list[str] = []
        otc_medicines: list[dict[str, object]] = []
        prescription_medicines: list[dict[str, object]] = []
        interaction_warnings: list[dict[str, object]] = []

        if urgent_matches:
            warnings = self.data_loader.get_warnings(
                patient_info=normalized_patient,
                urgent_matches=urgent_matches,
            )
        else:
            conditions, condition_terms = self._semantic_conditions(semantic_results)
            treatments = self._semantic_treatments(semantic_results)

            grouped_results = group_results_by_generic(semantic_results)
            all_medicines = self._semantic_medicine_entries(grouped_results, condition_terms)
            otc_medicines = [item for item in all_medicines if item["type"] == "OTC"]
            prescription_medicines = [item for item in all_medicines if item["type"] == "Prescription"]

            home_care = get_homecare_advice(
                conditions=condition_terms,
                category_map=self.condition_category_map,
                categories=self.homecare_categories,
                fallback=self._GENERIC_HOME_CARE,
            )

            warnings = []
            if not semantic_results:
                fallback_warning = self.data_loader.get_system_message(
                    "no_pattern_match",
                    "No clear condition detected. Please provide more detailed symptoms.",
                )
                if fallback_warning:
                    warnings.append(fallback_warning)
            if all_medicines:
                country_notice = self.data_loader.get_system_message("country_labeling")
                if country_notice:
                    warnings.append(country_notice)
            warnings = self._dedupe(warnings)

        if urgent_matches:
            home_care = []

        english_payload = {
            "symptoms": symptoms,
            "conditions": [
                {
                    "condition": item["condition"],
                    "confidence": item["confidence"],
                    "reason": item["reason"],
                }
                for item in conditions
            ],
            "treatments": treatments,
            "otc_medicines": self._public_medicine_entries(otc_medicines),
            "prescription_medicines": self._public_medicine_entries(prescription_medicines),
            "home_care": home_care,
            "warnings": self._dedupe(warnings),
            "interaction_warnings": interaction_warnings,
        }

        result = self._localize_payload(english_payload, user_lang, nllb_lang)
        self._cache_set(cache_key, result)
        return result
