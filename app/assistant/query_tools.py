import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.assistant.artifacts import ConditionArtifacts
from app.assistant.condition_groups import ConditionGroupNormalizer


@dataclass(frozen=True)
class DetectedGroup:
    group: str
    evidence: list[str]


def load_home_advice(path: Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized: dict[str, list[str]] = {}
    for key, value in (data or {}).items():
        items = [str(item).strip() for item in (value or []) if str(item).strip()]
        if items:
            normalized[str(key).strip().lower()] = items
    return normalized


def detect_condition_groups(
    query_en: str,
    artifacts: ConditionArtifacts,
    normalizer: ConditionGroupNormalizer,
    max_groups: int = 3,
) -> list[DetectedGroup]:
    normalized_query = normalizer.clean_phrase(query_en)
    if not normalized_query:
        return []

    phrase_to_group = artifacts.phrase_to_group()
    evidence_by_group: dict[str, list[str]] = {}

    tokens = normalized_query.split()
    candidates = {normalized_query}
    for window in range(1, min(5, len(tokens)) + 1):
        for idx in range(0, len(tokens) - window + 1):
            candidates.add(" ".join(tokens[idx:idx + window]))

    for phrase in sorted(candidates, key=len, reverse=True):
        group = phrase_to_group.get(phrase.lower())
        if not group:
            group = normalizer.to_group(phrase)
        if not group:
            continue

        # Avoid exploding into thousands of literal phrase groups.
        if group == phrase and group not in artifacts.groups:
            continue

        evidence_by_group.setdefault(group, []).append(phrase)

    detected = [
        DetectedGroup(group=group, evidence=sorted(set(evidence), key=len, reverse=True))
        for group, evidence in evidence_by_group.items()
    ]
    detected.sort(key=lambda item: (item.group not in artifacts.groups, len(item.evidence)), reverse=False)
    return detected[:max_groups]


def expand_query(
    query_en: str,
    groups: list[DetectedGroup],
    artifacts: ConditionArtifacts,
    max_variants_per_group: int = 20,
) -> str:
    base = re.sub(r"\s+", " ", (query_en or "").strip())
    if not base:
        return ""

    if not groups:
        return base

    terms: list[str] = [base]
    for item in groups:
        terms.append(item.group)
        terms.extend(artifacts.variants_for(item.group, limit=max_variants_per_group))

    return " ".join(dict.fromkeys(term.strip() for term in terms if term.strip()))


def group_results_by_generic(results: list[dict[str, object]], max_brands_per_generic: int = 6) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}

    for item in results:
        generic = str(item.get("generic") or "Unknown").strip()
        if not generic:
            continue

        entry = grouped.get(generic)
        if entry is None:
            entry = {
                "generic_name": generic,
                "category": item.get("category", ""),
                "uses": item.get("description", ""),
                "composition": item.get("composition", ""),
                "warnings": item.get("warnings", ""),
                "best_relevance": int(item.get("relevance", 0) or 0),
                "brands": [],
            }
            grouped[generic] = entry

        entry["best_relevance"] = max(entry["best_relevance"], int(item.get("relevance", 0) or 0))

        brands = item.get("brands") or []
        for brand in brands:
            if len(entry["brands"]) >= max_brands_per_generic:
                break
            name = str(brand.get("name", "")).strip()
            if not name:
                continue
            if any(existing.get("name") == name for existing in entry["brands"]):
                continue
            entry["brands"].append(brand)

    output = list(grouped.values())
    output.sort(key=lambda item: item.get("best_relevance", 0), reverse=True)
    return output
