import re


_LEADING_PHRASES = (
    "treatment of",
    "prevention of",
    "management of",
    "relief of",
    "used for",
    "used to treat",
    "used in",
    "therapy of",
    "indicated for",
    "for treatment of",
)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[\u2012-\u2015]", "-", lowered)
    lowered = re.sub(r"[^a-z0-9\s\-+/,&']", " ", lowered)
    lowered = re.sub(r"\b(paining|hurting|hurts|aching|ached|aches)\b", "pain", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _strip_leading_phrases(text: str) -> str:
    cleaned = text
    changed = True
    while changed:
        changed = False
        for phrase in _LEADING_PHRASES:
            if cleaned.startswith(phrase + " "):
                cleaned = cleaned[len(phrase):].strip()
                changed = True
    return cleaned


def _strip_trailing_explanations(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\bdue to\b.*$", "", cleaned).strip()
    cleaned = re.sub(r"\bassociated with\b.*$", "", cleaned).strip()
    cleaned = re.sub(r"\bcaused by\b.*$", "", cleaned).strip()
    cleaned = re.sub(r"\bresulting from\b.*$", "", cleaned).strip()
    return cleaned


def extract_conditions_from_uses(uses_text: str) -> list[str]:
    """
    Extract condition phrases from the dataset `uses` column.

    Examples:
      "Treatment of Bacterial infections" -> ["bacterial infections"]
      "Treatment of cough, Treatment of dry cough" -> ["cough", "dry cough"]

    Notes:
    - This is intentionally heuristic and dataset-driven.
    - Do not hardcode medical ontologies here; clustering handles variation.
    """
    normalized = _normalize_text(uses_text)
    if not normalized:
        return []

    normalized = _strip_leading_phrases(normalized)
    normalized = _strip_trailing_explanations(normalized)
    if not normalized:
        return []

    # Split into phrases. Keep "and" splitting conservative by only splitting on explicit separators.
    # The dataset often uses commas between "Treatment of ..." segments.
    separators = [",", ";", "|", "/", "+", "&"]
    parts = [normalized]
    for sep in separators:
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend([item.strip() for item in part.split(sep)])
        parts = next_parts

    # Split on " and " as a second pass (helps "sneezing and runny nose").
    second_pass: list[str] = []
    for part in parts:
        if " and " in part:
            second_pass.extend([item.strip() for item in part.split(" and ")])
        else:
            second_pass.append(part)
    parts = second_pass

    conditions: list[str] = []
    for part in parts:
        cleaned = _normalize_text(part)
        cleaned = _strip_leading_phrases(cleaned)
        cleaned = _strip_trailing_explanations(cleaned)
        cleaned = cleaned.strip(" -")
        if not cleaned:
            continue
        if len(cleaned) < 3:
            continue
        conditions.append(cleaned)

    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in conditions:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped

