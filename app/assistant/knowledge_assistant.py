import json
from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from app.config import LANG_MAP
from app.nlp.language import detect_language
from app.nlp.translator import Translator
from app.search.semantic_search import SemanticSearchEngine
from app.assistant.query_tools import group_results_by_generic, load_home_advice


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DISCLAIMER = (
    "Educational information only. This is not a prescription. "
    "Consult a qualified clinician for medical advice."
)


@dataclass(frozen=True)
class ConditionArtifacts:
    conditions: list[str]
    clusters: dict[str, list[str]]
    condition_to_cluster: dict[str, str]

    @classmethod
    def load(cls, index_dir: Path) -> "ConditionArtifacts":
        conditions_path = index_dir / "conditions.json"
        clusters_path = index_dir / "condition_clusters.json"
        mapping_path = index_dir / "condition_to_cluster.json"

        conditions = json.loads(conditions_path.read_text(encoding="utf-8"))
        clusters = json.loads(clusters_path.read_text(encoding="utf-8")) if clusters_path.exists() else {}
        condition_to_cluster = (
            json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
        )
        return cls(
            conditions=[str(item) for item in conditions if str(item).strip()],
            clusters={str(k): [str(v) for v in vals] for k, vals in clusters.items()},
            condition_to_cluster={str(k): str(v) for k, v in condition_to_cluster.items()},
        )


class ConditionSearcher:
    def __init__(self, index_dir: Path, model_name: str = DEFAULT_MODEL) -> None:
        index_path = index_dir / "conditions.index"
        if not index_path.exists():
            raise RuntimeError(f"Missing condition index at {index_path}")

        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name, device="cpu")

    def search(self, query: str, top_k: int = 6) -> list[tuple[int, float]]:
        if not query or not query.strip():
            return []

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(embedding, top_k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results


class MedicalKnowledgeAssistant:
    def __init__(
        self,
        index_dir: Path | None = None,
        data_dir: Path | None = None,
        model_name: str = DEFAULT_MODEL,
        min_condition_score: float = 0.35,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.index_dir = index_dir or (base_dir / "indexes")
        self.data_dir = data_dir or (base_dir / "data")
        self.model_name = model_name
        self.min_condition_score = min_condition_score

        self.translator = Translator()
        self.search_engine = SemanticSearchEngine()
        self.artifacts = ConditionArtifacts.load(self.index_dir)
        self.condition_searcher = ConditionSearcher(self.index_dir, model_name=model_name)
        self.home_advice = {}
        advice_path = self.data_dir / "home_advice.json"
        if advice_path.exists():
            self.home_advice = load_home_advice(advice_path)

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

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            key = str(item).strip().lower()
            if key and key not in seen:
                seen.add(key)
                output.append(str(item).strip())
        return output

    def _choose_cluster(
        self,
        scored_conditions: list[tuple[str, float]],
    ) -> str:
        if not scored_conditions:
            return ""
        cluster_scores: dict[str, float] = {}
        for condition, score in scored_conditions:
            if score < self.min_condition_score:
                continue
            cluster = self.artifacts.condition_to_cluster.get(condition)
            if not cluster:
                continue
            cluster_scores[cluster] = cluster_scores.get(cluster, 0.0) + score

        if not cluster_scores:
            return ""
        return max(cluster_scores.items(), key=lambda item: item[1])[0]

    def _expand_query(
        self,
        query_en: str,
        cluster: str,
        possible_conditions: list[str],
        max_variants: int = 12,
    ) -> str:
        terms: list[str] = [query_en]
        if cluster:
            terms.append(cluster)
            variants = self.artifacts.clusters.get(cluster, [])
            terms.extend(variants[:max_variants])
        terms.extend(possible_conditions[:max_variants])
        return " ".join(self._dedupe([term for term in terms if term.strip()]))

    def _home_care_for(self, cluster: str, possible_conditions: list[str]) -> list[str]:
        if not self.home_advice:
            return []
        keys = [cluster] if cluster else []
        keys.extend(possible_conditions)
        tips: list[str] = []
        for key in keys:
            if not key:
                continue
            advice = self.home_advice.get(key.lower())
            if advice:
                tips.extend(advice)
        return self._dedupe(tips)

    def _treatments_from_results(self, results: list[dict[str, object]], limit: int = 15) -> list[dict[str, object]]:
        treatments: list[dict[str, object]] = []
        for entry in results[:limit]:
            treatments.append(
                {
                    "generic_name": entry.get("generic_name", ""),
                    "brands": [item.get("name", "") for item in entry.get("brands", [])],
                    "uses": entry.get("uses", ""),
                    "warnings": entry.get("warnings", ""),
                }
            )
        return treatments

    def run(self, query: str, language_hint: str | None = None) -> dict[str, object]:
        query = (query or "").strip()
        if not query:
            return {
                "query": "",
                "condition_cluster": "",
                "possible_conditions": [],
                "treatments": [],
                "home_care": [],
                "disclaimer": DEFAULT_DISCLAIMER,
            }

        user_lang, nllb_lang = self._resolve_language(query, language_hint)
        query_en = self._to_english(query, user_lang, nllb_lang)

        raw_matches = self.condition_searcher.search(query_en, top_k=8)
        scored_conditions: list[tuple[str, float]] = []
        for idx, score in raw_matches:
            if idx < 0 or idx >= len(self.artifacts.conditions):
                continue
            scored_conditions.append((self.artifacts.conditions[idx], score))

        possible_conditions = [condition for condition, score in scored_conditions if score >= self.min_condition_score]
        cluster = self._choose_cluster(scored_conditions)
        expanded_query = self._expand_query(query_en, cluster, possible_conditions)

        medicine_results = self.search_engine.search(expanded_query, top_k=30)
        grouped = group_results_by_generic(medicine_results)
        treatments = self._treatments_from_results(grouped)
        home_care = self._home_care_for(cluster, possible_conditions)

        return {
            "query": query,
            "condition_cluster": cluster,
            "possible_conditions": possible_conditions,
            "treatments": treatments,
            "home_care": home_care,
            "disclaimer": DEFAULT_DISCLAIMER,
        }
