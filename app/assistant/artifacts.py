import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.assistant.condition_groups import ConditionGroupNormalizer


@dataclass(frozen=True)
class ConditionArtifacts:
    groups: dict[str, dict[str, object]]

    @classmethod
    def load(cls, path: Path) -> "ConditionArtifacts":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(groups=data.get("groups", {}))

    def phrase_to_group(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for group, payload in self.groups.items():
            for item in payload.get("variants", []):
                phrase = str(item.get("phrase", "")).strip().lower()
                if phrase:
                    mapping[phrase] = group
        return mapping

    def variants_for(self, group: str, limit: int = 25) -> list[str]:
        payload = self.groups.get(group, {})
        variants = payload.get("variants", [])
        phrases = [str(item.get("phrase", "")).strip() for item in variants if str(item.get("phrase", "")).strip()]
        return phrases[:limit]

    def top_generics_for(self, group: str, limit: int = 25) -> list[str]:
        payload = self.groups.get(group, {})
        generics = payload.get("top_generics", [])
        names = [str(item.get("generic_name", "")).strip() for item in generics if str(item.get("generic_name", "")).strip()]
        return names[:limit]


def build_condition_artifacts(
    csv_path: Path,
    rules_path: Path,
    out_path: Path,
    uses_column: str = "uses",
    generic_column: str = "generic_name",
    chunksize: int = 20000,
    max_variants_per_group: int = 200,
    max_generics_per_group: int = 100,
) -> None:
    normalizer = ConditionGroupNormalizer(rules_path)
    variant_counts: dict[str, Counter[str]] = defaultdict(Counter)
    generic_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk[uses_column] = chunk.get(uses_column, "").fillna("")
        chunk[generic_column] = chunk.get(generic_column, "").fillna("")

        for uses_text, generic_name in zip(chunk[uses_column].tolist(), chunk[generic_column].tolist()):
            uses_text = str(uses_text or "")
            generic_name = str(generic_name or "").strip()
            if not uses_text.strip():
                continue

            phrases = normalizer.split_uses(uses_text)
            for phrase in phrases:
                group = normalizer.to_group(phrase)
                if not group:
                    continue
                variant_counts[group][phrase] += 1
                if generic_name:
                    generic_counts[group][generic_name] += 1

    groups: dict[str, dict[str, object]] = {}
    for group in sorted(variant_counts.keys()):
        variants = [
            {"phrase": phrase, "count": int(count)}
            for phrase, count in variant_counts[group].most_common(max_variants_per_group)
        ]
        top_generics = [
            {"generic_name": name, "count": int(count)}
            for name, count in generic_counts[group].most_common(max_generics_per_group)
        ]
        groups[group] = {
            "variants": variants,
            "top_generics": top_generics,
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"groups": groups}, indent=2), encoding="utf-8")
