import json
import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path


class MedicalDataLoader:
    def __init__(self, data_dir: Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir is not None else base_dir / "data"
        self._loaded = False
        self.symptom_patterns: dict[str, list[str]] = {}
        self.conditions: list[dict[str, object]] = []
        self.conditions_by_key: dict[str, dict[str, object]] = {}
        self.treatments: dict[str, list[str]] = {}
        self.medicines: list[dict[str, object]] = []
        self.medicines_by_key: dict[str, dict[str, object]] = {}
        self.homecare: dict[str, list[str]] = {}
        self.condition_warnings: dict[str, list[str]] = {}
        self.patient_advisories: dict[str, str] = {}
        self.system_messages: dict[str, str] = {}
        self.interaction_rules: list[dict[str, object]] = []
        self.urgent_risk_rules: list[dict[str, object]] = []
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
        self._symptom_phrase_map: dict[str, str] = {}
        self._max_symptom_phrase_tokens = 1
        self._condition_terms_by_key: dict[str, list[str]] = {}

    def load_datasets(self) -> None:
        if self._loaded:
            return

        self.symptom_patterns = self._load_json("symptoms.json")
        self.conditions = self._load_json("conditions.json")
        self.conditions_by_key = {
            str(item["key"]): item
            for item in self.conditions
            if isinstance(item, dict) and item.get("key")
        }
        self.treatments = self._load_json("treatments.json")
        self.medicines = self._load_json("medicines.json")
        self.medicines_by_key = {
            str(item["key"]): item
            for item in self.medicines
            if isinstance(item, dict) and item.get("key")
        }
        self.homecare = self._load_json("homecare.json")
        warning_data = self._load_json("warnings.json")
        self.condition_warnings = warning_data.get("condition_warnings", {})
        self.patient_advisories = warning_data.get("patient_advisories", {})
        self.system_messages = warning_data.get("system_messages", {})
        self.interaction_rules = warning_data.get("interaction_rules", [])
        self.urgent_risk_rules = warning_data.get("urgent_risk_rules", [])
        self._compiled_patterns = self._build_patterns()
        self._symptom_phrase_map = self._build_symptom_phrase_map()
        self._max_symptom_phrase_tokens = self._build_max_symptom_phrase_tokens()
        self._condition_terms_by_key = self._build_condition_terms()
        self._loaded = True

    def extract_symptoms(self, text: str) -> list[str]:
        self.load_datasets()
        normalized = self._normalize_text(text)
        findings: list[str] = []
        for symptom, pattern in self._compiled_patterns:
            if pattern.search(normalized):
                findings.append(symptom)

        candidates = self._build_query_candidates(normalized)
        symptom_phrases = list(self._symptom_phrase_map.keys())
        for candidate in candidates:
            if candidate in self._symptom_phrase_map:
                findings.append(self._symptom_phrase_map[candidate])
                continue

            match = get_close_matches(candidate, symptom_phrases, n=1, cutoff=0.78)
            if match:
                findings.append(self._symptom_phrase_map[match[0]])

        return self._dedupe(findings)

    def detect_condition_query(self, text: str, limit: int = 5) -> list[dict[str, object]]:
        self.load_datasets()
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        if normalized in self._symptom_phrase_map:
            return []

        results: list[dict[str, object]] = []
        for profile in self.conditions:
            condition_key = str(profile["key"])
            best_score = 0.0
            best_term = ""

            for term in self._condition_terms_by_key.get(condition_key, []):
                if not term:
                    continue

                score = 0.0
                if normalized == term:
                    score = 0.99
                elif len(normalized) >= 4 and (normalized in term or term in normalized):
                    score = 0.92
                else:
                    ratio = SequenceMatcher(None, normalized, term).ratio()
                    if ratio >= 0.84:
                        score = min(0.95, ratio)

                if score > best_score:
                    best_score = score
                    best_term = term

            if best_score < 0.84:
                continue

            results.append(
                {
                    "condition": str(profile["name"]),
                    "condition_key": condition_key,
                    "confidence": round(best_score, 2),
                    "reason": self._direct_condition_reason(str(profile["name"]), best_term),
                    "_score": best_score,
                }
            )

        results.sort(key=lambda item: item["_score"], reverse=True)
        trimmed = results[:limit]
        for item in trimmed:
            item.pop("_score", None)
        return trimmed

    def get_conditions(self, symptoms: list[str], text: str = "", limit: int = 5) -> list[dict[str, object]]:
        self.load_datasets()
        normalized = self._normalize_text(text)
        symptom_set = set(self._canonicalize_symptoms(symptoms))
        if not symptom_set:
            return []

        results: list[dict[str, object]] = []

        for profile in self.conditions:
            symptom_weights = profile.get("symptom_weights", {})
            total_weight = sum(symptom_weights.values()) or 1.0
            matched_symptoms = [symptom for symptom in symptom_weights if symptom in symptom_set]
            if not matched_symptoms:
                continue

            matched_weight = sum(symptom_weights[symptom] for symptom in matched_symptoms)
            keywords = [str(keyword) for keyword in profile.get("keywords", [])]
            keyword_hit = any(keyword in normalized for keyword in keywords)
            keyword_bonus = 0.08 if keyword_hit else 0.0

            normalized_overlap = matched_weight / total_weight
            symptom_coverage = len(matched_symptoms) / len(symptom_set)
            symptom_bonus = min(0.18, len(matched_symptoms) * 0.06)
            raw_similarity = matched_weight / max(1, len(symptom_set))
            confidence = min(
                0.95,
                0.1
                + (raw_similarity * 0.45)
                + (normalized_overlap * 0.2)
                + (symptom_coverage * 0.15)
                + symptom_bonus
                + keyword_bonus,
            )
            if confidence < 0.5:
                continue

            results.append(
                {
                    "condition": str(profile["name"]),
                    "condition_key": str(profile["key"]),
                    "confidence": round(confidence, 2),
                    "reason": self._condition_reason(str(profile["name"]), matched_symptoms, keyword_hit),
                    "_score": confidence,
                    "_match_count": len(matched_symptoms),
                }
            )

        results.sort(key=lambda item: (item["_score"], item["_match_count"]), reverse=True)
        trimmed = self.validate_condition_results(results[:limit], list(symptom_set))
        for item in trimmed:
            item.pop("_score", None)
            item.pop("_match_count", None)
        return trimmed

    def get_treatments(self, condition_key: str) -> list[str]:
        self.load_datasets()
        return list(self.treatments.get(condition_key, []))

    def get_homecare(self, condition_key: str) -> list[str]:
        self.load_datasets()
        return list(self.homecare.get(condition_key, []))

    def get_condition_warnings(self, condition_key: str) -> list[str]:
        self.load_datasets()
        return list(self.condition_warnings.get(condition_key, []))

    def get_condition_symptoms(self, condition_key: str) -> list[str]:
        self.load_datasets()
        condition = self.conditions_by_key.get(condition_key)
        if not condition:
            return []
        symptom_weights = condition.get("symptom_weights", {})
        ordered = sorted(symptom_weights.items(), key=lambda item: item[1], reverse=True)
        return [str(symptom) for symptom, _ in ordered]

    def get_related_condition_keys(
        self,
        symptoms: list[str],
        exclude_condition_keys: list[str] | None = None,
        limit: int = 3,
    ) -> list[str]:
        self.load_datasets()
        symptom_set = set(self._canonicalize_symptoms(symptoms))
        exclude = set(exclude_condition_keys or [])
        if not symptom_set:
            return []

        scored: list[tuple[float, str]] = []
        for profile in self.conditions:
            condition_key = str(profile["key"])
            if condition_key in exclude:
                continue

            symptom_weights = profile.get("symptom_weights", {})
            matched_symptoms = [symptom for symptom in symptom_weights if symptom in symptom_set]
            if not matched_symptoms:
                continue

            score = sum(symptom_weights[symptom] for symptom in matched_symptoms)
            scored.append((score, condition_key))

        scored.sort(reverse=True)
        return [condition_key for _, condition_key in scored[:limit]]

    def get_medicines(
        self,
        condition_keys: list[str],
        patient_info: dict[str, object] | None = None,
        query_text: str = "",
    ) -> dict[str, list[dict[str, object]] | list[str]]:
        self.load_datasets()
        patient_info = patient_info or {}
        selected_keys: list[str] = []
        seen: set[str] = set()

        for condition_key in condition_keys[:3]:
            for medicine in self.medicines:
                medicine_key = str(medicine.get("key", ""))
                used_for = [str(item) for item in medicine.get("used_for", [])]
                if condition_key in used_for and medicine_key and medicine_key not in seen:
                    seen.add(medicine_key)
                    selected_keys.append(medicine_key)

        for mentioned_key in self.mentioned_medicine_keys(query_text):
            if mentioned_key not in seen:
                seen.add(mentioned_key)
                selected_keys.append(mentioned_key)

        medicines: list[dict[str, object]] = []
        warnings: list[str] = []
        grouped_conditions = {
            condition_key: str(self.conditions_by_key[condition_key]["name"])
            for condition_key in condition_keys
            if condition_key in self.conditions_by_key
        }

        for key in selected_keys:
            medicine = self.medicines_by_key.get(key)
            if medicine is None:
                continue

            patient_warning = self._patient_warning(medicine, patient_info)
            if patient_warning and patient_warning.startswith("Avoid"):
                warnings.append(f"{medicine['name']}: {patient_warning}")
                continue

            related_conditions = [
                grouped_conditions[condition_key]
                for condition_key in medicine.get("used_for", [])
                if condition_key in grouped_conditions
            ]

            medicines.append(
                {
                    "name": str(medicine["name"]),
                    "type": str(medicine["classification"]),
                    "purpose": str(medicine["purpose"]),
                    "safety_notes": list(medicine.get("safety_notes", [])),
                    "related_conditions": related_conditions,
                    "patient_specific_warning": patient_warning or "",
                    "medicine_key": str(medicine["key"]),
                }
            )

        medicines.sort(key=lambda item: (item["type"] != "OTC", item["name"]))
        return {
            "all_medicines": medicines,
            "otc_medicines": [item for item in medicines if item["type"] == "OTC"],
            "prescription_medicines": [item for item in medicines if item["type"] == "Prescription"],
            "warnings": self._dedupe(warnings),
        }

    def get_warnings(
        self,
        patient_info: dict[str, object] | None = None,
        condition_keys: list[str] | None = None,
        urgent_matches: list[dict[str, str]] | None = None,
        medicine_warnings: list[str] | None = None,
        no_pattern_match: bool = False,
        include_country_notice: bool = False,
    ) -> list[str]:
        self.load_datasets()
        patient_info = patient_info or {}
        warnings: list[str] = []

        if urgent_matches:
            banner = self.get_system_message("urgent_banner")
            action = self.get_system_message("urgent_action")
            if banner:
                warnings.append(banner)
            if action:
                warnings.append(action)
            warnings.extend(
                [f"{match['label']}: {match['message']}" for match in urgent_matches]
            )

        age = patient_info.get("age")
        pregnancy_status = str(patient_info.get("pregnancy_status") or "").strip().lower()
        allergies = patient_info.get("allergies", [])

        if pregnancy_status == "pregnant" and self.patient_advisories.get("pregnant"):
            warnings.append(self.patient_advisories["pregnant"])
        if isinstance(age, int) and age < 12 and self.patient_advisories.get("child"):
            warnings.append(self.patient_advisories["child"])
        if allergies and self.patient_advisories.get("allergies_present"):
            warnings.append(self.patient_advisories["allergies_present"])

        if not urgent_matches:
            for condition_key in condition_keys or []:
                warnings.extend(self.get_condition_warnings(condition_key))
            warnings.extend(medicine_warnings or [])
            if no_pattern_match and self.system_messages.get("no_pattern_match"):
                warnings.append(self.system_messages["no_pattern_match"])
            if include_country_notice and self.system_messages.get("country_labeling"):
                warnings.append(self.system_messages["country_labeling"])

        return self._dedupe(warnings)

    def get_interaction_warnings(
        self,
        medicine_entries: list[dict[str, object]],
        patient_info: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.load_datasets()
        patient_info = patient_info or {}
        selected_keys = {
            str(entry.get("medicine_key", "")).strip()
            for entry in medicine_entries
            if entry.get("medicine_key")
        }

        current_medications = {
            self._normalize_text(str(value))
            for value in patient_info.get("current_medications", [])
            if str(value).strip()
        }
        for key, record in self.medicines_by_key.items():
            aliases = {
                self._normalize_text(record["name"]),
                *(self._normalize_text(alias) for alias in record.get("aliases", [])),
            }
            if any(alias in current_medications for alias in aliases):
                selected_keys.add(key)

        warnings: list[dict[str, object]] = []
        for rule in self.interaction_rules:
            medicine_keys = [str(item) for item in rule.get("medicine_keys", [])]
            if len(medicine_keys) != 2:
                continue
            left, right = medicine_keys
            if left in selected_keys and right in selected_keys:
                warnings.append(
                    {
                        "medicines": [
                            str(self.medicines_by_key[left]["name"]),
                            str(self.medicines_by_key[right]["name"]),
                        ],
                        "reason": str(rule["reason"]),
                    }
                )
        return warnings

    def detect_urgent_risks(self, text: str) -> list[dict[str, str]]:
        self.load_datasets()
        normalized = self._normalize_text(text)
        matches: list[dict[str, str]] = []

        for rule in self.urgent_risk_rules:
            patterns = [str(pattern) for pattern in rule.get("patterns", [])]
            if any(pattern in normalized for pattern in patterns):
                matches.append(
                    {
                        "label": str(rule["label"]),
                        "message": str(rule["message"]),
                    }
                )

        return matches

    def get_system_message(self, key: str, default: str = "") -> str:
        self.load_datasets()
        return str(self.system_messages.get(key, default))

    def mentioned_medicine_keys(self, text: str) -> list[str]:
        self.load_datasets()
        normalized = self._normalize_text(text)
        matches: list[str] = []
        for key, medicine in self.medicines_by_key.items():
            aliases = {
                self._normalize_text(medicine["name"]),
                *(self._normalize_text(alias) for alias in medicine.get("aliases", [])),
            }
            if any(alias in normalized for alias in aliases):
                matches.append(key)
        return matches

    def _load_json(self, filename: str):
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for symptom, phrases in self.symptom_patterns.items():
            normalized_phrases = [self._normalize_text(str(phrase)) for phrase in phrases]
            ordered_phrases = sorted(normalized_phrases, key=len, reverse=True)
            pattern = r"\b(?:%s)\b" % "|".join(re.escape(phrase) for phrase in ordered_phrases)
            compiled.append((str(symptom), re.compile(pattern)))
        return compiled

    def _build_symptom_phrase_map(self) -> dict[str, str]:
        phrase_map: dict[str, str] = {}
        for symptom, phrases in self.symptom_patterns.items():
            normalized_symptom = self._normalize_text(str(symptom))
            if normalized_symptom:
                phrase_map[normalized_symptom] = str(symptom)
            for phrase in phrases:
                normalized_phrase = self._normalize_text(str(phrase))
                if normalized_phrase:
                    phrase_map[normalized_phrase] = str(symptom)
        return phrase_map

    def _build_max_symptom_phrase_tokens(self) -> int:
        max_tokens = 1
        for phrase in self._symptom_phrase_map:
            max_tokens = max(max_tokens, len(phrase.split()))
        return max_tokens

    def _build_condition_terms(self) -> dict[str, list[str]]:
        condition_terms: dict[str, list[str]] = {}
        for profile in self.conditions:
            key = str(profile["key"])
            terms = {
                self._normalize_text(str(profile["name"])),
                self._normalize_text(str(profile["key"]).replace("_", " ")),
            }
            for keyword in profile.get("keywords", []):
                terms.add(self._normalize_text(str(keyword)))
            condition_terms[key] = sorted(term for term in terms if term)
        return condition_terms

    def _build_query_candidates(self, normalized_text: str) -> list[str]:
        tokens = normalized_text.split()
        candidates: list[str] = [normalized_text] if normalized_text else []
        max_window = min(len(tokens), self._max_symptom_phrase_tokens)
        for window in range(1, max_window + 1):
            for index in range(0, len(tokens) - window + 1):
                candidate = " ".join(tokens[index:index + window]).strip()
                if candidate:
                    candidates.append(candidate)
        return self._dedupe(candidates)

    def _canonicalize_symptoms(self, symptoms: list[str]) -> list[str]:
        known_symptoms = {
            self._normalize_text(symptom): symptom
            for symptom in self.symptom_patterns
        }
        canonical: list[str] = []
        for symptom in symptoms:
            normalized = self._normalize_text(symptom)
            if normalized in known_symptoms:
                canonical.append(known_symptoms[normalized])
                continue

            match = get_close_matches(normalized, list(known_symptoms.keys()), n=1, cutoff=0.6)
            if match:
                canonical.append(known_symptoms[match[0]])

        return self._dedupe(canonical)

    def validate_condition_results(
        self,
        conditions: list[dict[str, object]],
        symptoms: list[str],
    ) -> list[dict[str, object]]:
        self.load_datasets()
        symptom_set = set(self._canonicalize_symptoms(symptoms))
        validated: list[dict[str, object]] = []
        for condition in conditions:
            condition_key = str(condition.get("condition_key", ""))
            profile = self.conditions_by_key.get(condition_key)
            if profile is None:
                continue
            if float(condition.get("confidence", 0.0)) < 0.5:
                continue
            profile_symptoms = set(profile.get("symptom_weights", {}))
            if symptom_set and not (symptom_set & profile_symptoms):
                continue
            validated.append(condition)
        return validated

    def validate_medicine_entries(
        self,
        medicines: list[dict[str, object]],
        allowed_condition_keys: list[str],
    ) -> list[dict[str, object]]:
        self.load_datasets()
        if not allowed_condition_keys:
            return medicines

        allowed = set(allowed_condition_keys)
        validated: list[dict[str, object]] = []
        for medicine in medicines:
            medicine_key = str(medicine.get("medicine_key", ""))
            record = self.medicines_by_key.get(medicine_key)
            if record is None:
                continue
            used_for = {str(item) for item in record.get("used_for", [])}
            if used_for & allowed:
                validated.append(medicine)
        return validated

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = (text or "").lower()
        lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
        lowered = re.sub(r"\b(paining|hurting|hurts|aching|ached|aches)\b", "pain", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _normalize_terms(values: list[str]) -> set[str]:
        normalized: set[str] = set()
        for value in values:
            text = re.sub(r"\s+", " ", (value or "").strip().lower())
            if text:
                normalized.add(text)
        return normalized

    @staticmethod
    def _contains_match(source_values: set[str], target_values: list[str]) -> bool:
        for source in source_values:
            for target in target_values:
                if target in source or source in target:
                    return True
        return False

    @staticmethod
    def _condition_reason(condition_name: str, matched_symptoms: list[str], keyword_hit: bool) -> str:
        if matched_symptoms:
            highlighted = ", ".join(matched_symptoms[:3])
            return f"{highlighted.capitalize()} can be seen with {condition_name.lower()}."
        if keyword_hit:
            return f"The query directly mentions terms related to {condition_name.lower()}."
        return f"The pattern has partial overlap with symptoms often seen in {condition_name.lower()}."

    @staticmethod
    def _direct_condition_reason(condition_name: str, matched_term: str) -> str:
        if matched_term:
            return f"The query directly matches {condition_name.lower()}."
        return f"The query directly mentions terms related to {condition_name.lower()}."

    def _patient_warning(self, medicine: dict[str, object], patient_info: dict[str, object]) -> str | None:
        age = patient_info.get("age")
        pregnancy_status = str(patient_info.get("pregnancy_status") or "").strip().lower()
        allergies = self._normalize_terms(patient_info.get("allergies", []))
        existing_conditions = self._normalize_terms(patient_info.get("existing_conditions", []))

        min_age = medicine.get("min_age")
        if isinstance(age, int) and min_age is not None and age < int(min_age):
            return f"Not usually used without clinician guidance below age {min_age}."

        if medicine.get("avoid_pregnancy") and pregnancy_status == "pregnant":
            return "Avoided in pregnancy unless a clinician says it is necessary."

        if allergies and self._contains_match(allergies, list(medicine.get("avoid_allergy_tags", []))):
            return "Avoid because the listed allergies may conflict with this medicine."

        if existing_conditions and self._contains_match(existing_conditions, list(medicine.get("avoid_conditions", []))):
            return "Avoid because one of the listed existing conditions increases safety risk."

        if existing_conditions and self._contains_match(existing_conditions, list(medicine.get("caution_conditions", []))):
            return "Use caution because one of the listed existing conditions can change suitability."

        return None

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


_DEFAULT_LOADER: MedicalDataLoader | None = None


def get_data_loader(data_dir: Path | None = None) -> MedicalDataLoader:
    global _DEFAULT_LOADER
    if _DEFAULT_LOADER is None or (data_dir is not None and Path(data_dir) != _DEFAULT_LOADER.data_dir):
        _DEFAULT_LOADER = MedicalDataLoader(data_dir)
    _DEFAULT_LOADER.load_datasets()
    return _DEFAULT_LOADER
