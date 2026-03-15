import faiss
import pickle
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearchEngine:
    _QUERY_STOPWORDS = {
        "for", "with", "and", "the", "a", "an", "of", "to", "in", "on",
        "need", "want", "medicine", "medicines", "treatment", "help",
        "from", "due", "my", "is", "are"
    }

    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "indexes"

        index_path = index_dir / "medicine.index"
        meta_path = index_dir / "medicine_metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise RuntimeError("FAISS index or metadata file missing")

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

    @staticmethod
    def _to_text(value):
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    @staticmethod
    def _score_to_percent(score: float) -> int:
        # Index is cosine-like (normalized vectors with inner product).
        pct = int(round(((score + 1.0) / 2.0) * 100.0))
        return max(1, min(99, pct))

    @classmethod
    def _query_terms(cls, query: str):
        terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        return {t for t in terms if t not in cls._QUERY_STOPWORDS}

    @staticmethod
    def _lexical_bonus(query_terms, candidate_text: str) -> int:
        if not query_terms:
            return 0
        candidate_terms = set(re.findall(r"[a-zA-Z]{3,}", candidate_text.lower()))
        overlap = len(query_terms.intersection(candidate_terms))
        return min(20, overlap * 4)

    @staticmethod
    def _confidence_bucket(relevance_score: int) -> str:
        if relevance_score >= 80:
            return "high"
        if relevance_score >= 60:
            return "medium"
        return "low"

    @staticmethod
    def _infer_form(pack_size: str, brand_name: str) -> str:
        text = f"{pack_size} {brand_name}".lower()
        if "injection" in text or "vial" in text:
            return "Injection"
        if "syrup" in text:
            return "Syrup"
        if "capsule" in text:
            return "Capsule"
        if "gel" in text:
            return "Gel"
        if "ointment" in text:
            return "Ointment"
        if "suspension" in text:
            return "Suspension"
        if "spray" in text:
            return "Spray"
        if "drop" in text:
            return "Drops"
        return "Tablet"

    @staticmethod
    def _extract_strength(generic: str, composition: str) -> str:
        for source in (generic, composition):
            match = re.search(r"\(([^)]+)\)", source)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _parse_substitutes(substitutes_text: str):
        if not substitutes_text or substitutes_text.lower() == "not available":
            return []
        parts = [part.strip() for part in substitutes_text.split(",")]
        return [p for p in parts if p]

    def search(self, query: str, top_k: int = 30):
        if not query or not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        if self.index.d != query_embedding.shape[1]:
            raise RuntimeError(
                f"FAISS index dim {self.index.d} does not match embedding dim {query_embedding.shape[1]}"
            )

        fetch_k = min(len(self.metadata), max(top_k * 4, top_k))
        scores, indices = self.index.search(query_embedding, fetch_k)
        query_terms = self._query_terms(query)

        results = []
        for faiss_idx, score in zip(indices[0], scores[0]):
            if faiss_idx == -1:
                continue

            item = self.metadata.iloc[int(faiss_idx)]

            brand_name = self._to_text(item.get("brand_name"))
            generic_name = self._to_text(item.get("generic_name"))
            uses = self._to_text(item.get("uses"))
            composition = self._to_text(item.get("composition"))
            pack_size = self._to_text(item.get("pack_size"))
            side_effects = self._to_text(item.get("warnings"))
            substitutes_text = self._to_text(item.get("substitutes"))
            category = self._to_text(item.get("medicine_type")) or "Medicine"

            base_relevance = self._score_to_percent(float(score))
            bonus = self._lexical_bonus(
                query_terms,
                " ".join([brand_name, generic_name, composition, uses])
            )
            relevance = max(1, min(99, base_relevance + bonus))
            confidence = self._confidence_bucket(relevance)
            form = self._infer_form(pack_size, brand_name)
            strength = self._extract_strength(generic_name, composition)

            brands = []
            if brand_name:
                brands.append({
                    "name": brand_name,
                    "form": form,
                    "strength": strength,
                    "confidence": confidence
                })

            for substitute in self._parse_substitutes(substitutes_text)[:3]:
                brands.append({
                    "name": substitute,
                    "form": form,
                    "strength": strength,
                    "confidence": "medium"
                })

            results.append({
                "generic": generic_name or "Unknown",
                "category": category,
                "description": uses or "Used for symptom relief",
                "composition": composition,
                "warnings": side_effects,
                "relevance": relevance,
                "brands": brands,
            })

        results.sort(key=lambda item: item.get("relevance", 0), reverse=True)
        return results[:top_k]
