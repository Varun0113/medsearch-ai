import json
import re
from pathlib import Path


def _normalize_key(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def load_homecare_categories(path: Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized: dict[str, list[str]] = {}
    for key, value in (data or {}).items():
        category = _normalize_key(str(key))
        items = [str(item).strip() for item in (value or []) if str(item).strip()]
        if category and items:
            normalized[category] = items
    return normalized


def load_condition_category_map(path: Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized: dict[str, str] = {}
    for condition, category in (data or {}).items():
        condition_key = _normalize_key(str(condition))
        category_key = _normalize_key(str(category))
        if condition_key and category_key:
            normalized[condition_key] = category_key
    return normalized


def get_homecare_advice(
    conditions: list[str],
    category_map: dict[str, str],
    categories: dict[str, list[str]],
    fallback: str,
) -> list[str]:
    if not conditions:
        return [fallback]

    advice: list[str] = []
    for condition in conditions:
        condition_key = _normalize_key(condition)
        category = category_map.get(condition_key)
        if not category:
            continue
        advice.extend(categories.get(category, []))

    seen: set[str] = set()
    deduped: list[str] = []
    for item in advice:
        key = _normalize_key(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped if deduped else [fallback]
