import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans

from app.assistant.uses_extraction import extract_conditions_from_uses


DEFAULT_MODEL = "all-MiniLM-L6-v2"
MAX_AGGLOMERATIVE = 5000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() == "nan":
        return ""
    return text


def build_condition_counts(
    csv_path: Path,
    uses_column: str,
    chunksize: int,
) -> Counter:
    counts: Counter = Counter()
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if uses_column not in chunk.columns:
            raise ValueError(f"Missing column: {uses_column}")

        uses_series = chunk[uses_column].fillna("")
        for uses_text in uses_series.tolist():
            for condition in extract_conditions_from_uses(str(uses_text or "")):
                counts[condition] += 1

        total_rows += len(chunk)
        if total_rows % (chunksize * 5) == 0:
            log(f"Processed {total_rows} rows for condition extraction")

    log(f"Condition phrases extracted: {len(counts)}")
    return counts


def encode_conditions(
    model: SentenceTransformer,
    conditions: list[str],
    batch_size: int,
) -> np.ndarray:
    if not conditions:
        return np.zeros((0, 0), dtype="float32")

    embeddings: list[np.ndarray] = []
    for start in range(0, len(conditions), batch_size):
        batch = conditions[start:start + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.append(batch_embeddings)

    merged = np.vstack(embeddings).astype("float32")
    return merged


def build_condition_index(
    embeddings: np.ndarray,
    out_path: Path,
) -> None:
    if embeddings.size == 0:
        raise ValueError("No embeddings available to build condition index")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))


def _agglomerative_labels(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="cosine",
            linkage="average",
        )
    return clusterer.fit_predict(embeddings)


def _kmeans_labels(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    n_clusters = max(2, min(n_clusters, embeddings.shape[0]))
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return clusterer.fit_predict(embeddings)


def build_condition_clusters(
    conditions: list[str],
    counts: Counter,
    embeddings: np.ndarray,
    method: str,
    distance_threshold: float,
    kmeans_clusters: int | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    if not conditions:
        return {}, {}

    if len(conditions) == 1:
        only = conditions[0]
        return {only: [only]}, {only: only}

    method = method.lower().strip()
    if method not in {"agglomerative", "kmeans"}:
        raise ValueError("cluster method must be 'agglomerative' or 'kmeans'")

    if method == "agglomerative" and len(conditions) > MAX_AGGLOMERATIVE:
        method = "kmeans"

    if method == "agglomerative":
        labels = _agglomerative_labels(embeddings, distance_threshold)
    else:
        if not kmeans_clusters:
            kmeans_clusters = max(20, int(len(conditions) ** 0.5))
        labels = _kmeans_labels(embeddings, kmeans_clusters)

    clusters_by_id: dict[int, list[str]] = defaultdict(list)
    for condition, label in zip(conditions, labels):
        clusters_by_id[int(label)].append(condition)

    cluster_map: dict[str, list[str]] = {}
    condition_to_cluster: dict[str, str] = {}
    used_labels: Counter = Counter()

    for label_id, cluster_conditions in clusters_by_id.items():
        cluster_conditions.sort(key=lambda c: (-counts.get(c, 0), len(c), c))
        base_label = cluster_conditions[0]
        suffix = used_labels[base_label]
        used_labels[base_label] += 1
        label = base_label if suffix == 0 else f"{base_label}_{suffix + 1}"
        cluster_map[label] = cluster_conditions
        for condition in cluster_conditions:
            condition_to_cluster[condition] = label

    return cluster_map, condition_to_cluster


def build_treatment_groups(
    csv_path: Path,
    uses_column: str,
    generic_column: str,
    brand_column: str,
    warnings_column: str,
    chunksize: int,
    max_brands: int,
    max_uses: int,
    max_warnings: int,
) -> dict[str, dict[str, list[str]]]:
    brand_counts: dict[str, Counter] = defaultdict(Counter)
    condition_counts: dict[str, Counter] = defaultdict(Counter)
    warning_counts: dict[str, Counter] = defaultdict(Counter)

    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for column in (uses_column, generic_column, brand_column, warnings_column):
            if column not in chunk.columns:
                raise ValueError(f"Missing column: {column}")

        uses_series = chunk[uses_column].fillna("")
        generic_series = chunk[generic_column].fillna("")
        brand_series = chunk[brand_column].fillna("")
        warnings_series = chunk[warnings_column].fillna("")

        for uses_text, generic_name, brand_name, warning_text in zip(
            uses_series.tolist(),
            generic_series.tolist(),
            brand_series.tolist(),
            warnings_series.tolist(),
        ):
            generic = normalize_text(generic_name)
            if not generic:
                continue

            brand = normalize_text(brand_name)
            if brand:
                brand_counts[generic][brand] += 1

            for condition in extract_conditions_from_uses(str(uses_text or "")):
                condition_counts[generic][condition] += 1

            warning = normalize_text(warning_text)
            if warning:
                warning_counts[generic][warning] += 1

        total_rows += len(chunk)
        if total_rows % (chunksize * 5) == 0:
            log(f"Processed {total_rows} rows for treatment groups")

    treatment_groups: dict[str, dict[str, list[str]]] = {}
    for generic in sorted(brand_counts.keys()):
        treatment_groups[generic] = {
            "brands": [name for name, _ in brand_counts[generic].most_common(max_brands)],
            "uses": [name for name, _ in condition_counts[generic].most_common(max_uses)],
            "warnings": [text for text, _ in warning_counts[generic].most_common(max_warnings)],
        }

    return treatment_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Build medical knowledge artifacts from the medicines CSV.")
    parser.add_argument("--csv", default="app/data/medicines_final.csv", help="Path to medicines CSV")
    parser.add_argument("--out-dir", default="indexes", help="Output directory for artifacts")
    parser.add_argument("--uses-column", default="uses", help="CSV column containing uses/indications text")
    parser.add_argument("--generic-column", default="generic_name", help="CSV column containing generic name")
    parser.add_argument("--brand-column", default="brand_name", help="CSV column containing brand name")
    parser.add_argument("--warnings-column", default="warnings", help="CSV column containing warnings text")
    parser.add_argument("--chunksize", type=int, default=20000, help="CSV chunk size")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformer model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--cluster-method", default="agglomerative", help="agglomerative or kmeans")
    parser.add_argument("--cluster-distance", type=float, default=0.35, help="Agglomerative distance threshold")
    parser.add_argument("--kmeans-clusters", type=int, default=0, help="KMeans cluster count (0=auto)")
    parser.add_argument("--max-brands", type=int, default=20, help="Max brands per generic")
    parser.add_argument("--max-uses", type=int, default=20, help="Max uses per generic")
    parser.add_argument("--max-warnings", type=int, default=6, help="Max warnings per generic")
    parser.add_argument("--skip-index", action="store_true", help="Skip building the conditions FAISS index")
    parser.add_argument("--skip-clusters", action="store_true", help="Skip condition clustering")
    parser.add_argument("--skip-treatments", action="store_true", help="Skip building treatment groups")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Extracting condition phrases...")
    condition_counts = build_condition_counts(csv_path, args.uses_column, args.chunksize)
    conditions = [condition for condition, _ in condition_counts.most_common()]

    conditions_path = out_dir / "conditions.json"
    write_json(conditions_path, conditions)
    log(f"Wrote {len(conditions)} conditions to {conditions_path}")

    embeddings = None
    if not args.skip_index or not args.skip_clusters:
        log("Loading embedding model...")
        model = SentenceTransformer(args.model, device="cpu")
        log("Encoding conditions...")
        embeddings = encode_conditions(model, conditions, args.batch_size)
        log(f"Condition embeddings shape: {embeddings.shape}")

    if not args.skip_index:
        index_path = out_dir / "conditions.index"
        log("Building FAISS condition index...")
        build_condition_index(embeddings, index_path)
        log(f"Wrote FAISS index to {index_path}")

    if not args.skip_clusters:
        if embeddings is None:
            raise ValueError("Embeddings are required for clustering")
        log("Clustering condition phrases...")
        cluster_map, condition_to_cluster = build_condition_clusters(
            conditions=conditions,
            counts=condition_counts,
            embeddings=embeddings,
            method=args.cluster_method,
            distance_threshold=args.cluster_distance,
            kmeans_clusters=args.kmeans_clusters or None,
        )
        clusters_path = out_dir / "condition_clusters.json"
        write_json(clusters_path, cluster_map)
        mapping_path = out_dir / "condition_to_cluster.json"
        write_json(mapping_path, condition_to_cluster)
        log(f"Wrote {len(cluster_map)} condition clusters to {clusters_path}")

    if not args.skip_treatments:
        log("Building treatment groups by generic name...")
        treatment_groups = build_treatment_groups(
            csv_path=csv_path,
            uses_column=args.uses_column,
            generic_column=args.generic_column,
            brand_column=args.brand_column,
            warnings_column=args.warnings_column,
            chunksize=args.chunksize,
            max_brands=args.max_brands,
            max_uses=args.max_uses,
            max_warnings=args.max_warnings,
        )
        treatments_path = out_dir / "treatment_groups.json"
        write_json(treatments_path, treatment_groups)
        log(f"Wrote {len(treatment_groups)} treatment groups to {treatments_path}")

    log("Medical knowledge build complete.")


if __name__ == "__main__":
    main()
