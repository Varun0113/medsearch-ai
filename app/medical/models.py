from dataclasses import dataclass


@dataclass(frozen=True)
class MedicineRecord:
    key: str
    name: str
    classification: str
    purpose: str
    safety_notes: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    used_for: tuple[str, ...] = ()
    drug_classes: tuple[str, ...] = ()
    min_age: int | None = None
    avoid_pregnancy: bool = False
    avoid_allergy_tags: tuple[str, ...] = ()
    avoid_conditions: tuple[str, ...] = ()
    caution_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConditionProfile:
    key: str
    name: str
    symptom_weights: dict[str, float]
    keywords: tuple[str, ...]
    treatment_options: tuple[str, ...]
    medicine_keys: tuple[str, ...]
    home_care: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class InteractionRule:
    medicine_keys: tuple[str, str]
    reason: str


@dataclass(frozen=True)
class RiskRule:
    label: str
    patterns: tuple[str, ...]
    message: str
