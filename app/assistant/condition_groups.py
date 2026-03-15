import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GroupRule:
    key: str
    priority: int
    patterns: tuple[re.Pattern[str], ...]


class ConditionGroupNormalizer:
    def __init__(self, rules_path: Path) -> None:
        self.rules_path = Path(rules_path)
        self._rules: list[GroupRule] = []
        self._strip_prefixes: list[str] = []
        self._split_separators: list[str] = []
        self._load()

    def _load(self) -> None:
        data = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self._strip_prefixes = [str(item).strip().lower() for item in data.get("strip_prefixes", []) if str(item).strip()]
        self._split_separators = [str(item) for item in data.get("split_separators", []) if str(item)]

        rules: list[GroupRule] = []
        groups = data.get("groups", {})
        for key, spec in groups.items():
            priority = int(spec.get("priority", 0))
            patterns = []
            for pattern in spec.get("patterns", []):
                source = str(pattern).strip()
                if not source:
                    continue
                patterns.append(re.compile(source, re.IGNORECASE))
            rules.append(GroupRule(key=str(key), priority=priority, patterns=tuple(patterns)))

        rules.sort(key=lambda item: item.priority, reverse=True)
        self._rules = rules

    @staticmethod
    def _basic_normalize(text: str) -> str:
        lowered = (text or "").lower()
        lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
        lowered = re.sub(r"\b(paining|hurting|hurts|aching|ached|aches)\b", "pain", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def clean_phrase(self, phrase: str) -> str:
        cleaned = self._basic_normalize(phrase)
        if not cleaned:
            return ""

        for prefix in self._strip_prefixes:
            if cleaned.startswith(prefix + " "):
                cleaned = cleaned[len(prefix):].strip()

        cleaned = re.sub(r"\bdue to\b.*$", "", cleaned).strip()
        cleaned = re.sub(r"\bassociated with\b.*$", "", cleaned).strip()
        cleaned = re.sub(r"\bcaused by\b.*$", "", cleaned).strip()
        return cleaned

    def split_uses(self, uses_text: str) -> list[str]:
        cleaned = self.clean_phrase(uses_text)
        if not cleaned:
            return []

        parts: list[str] = [cleaned]
        for sep in self._split_separators:
            next_parts: list[str] = []
            for part in parts:
                next_parts.extend([item.strip() for item in part.split(sep)])
            parts = next_parts

        results: list[str] = []
        for part in parts:
            normalized = self.clean_phrase(part)
            if normalized:
                results.append(normalized)
        return results

    def to_group(self, phrase: str) -> str:
        cleaned = self.clean_phrase(phrase)
        if not cleaned:
            return ""

        for rule in self._rules:
            for pattern in rule.patterns:
                if pattern.search(cleaned):
                    return rule.key
        return cleaned
