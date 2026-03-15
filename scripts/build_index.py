import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

DATA_PATH = Path("app/data/medicines_final.csv")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256          # CPU-optimized
CHUNK_SIZE = 20000        # Safe chunk size
EMBED_FIELDS = [
    "generic_name",
    "composition",
    "uses",
]

# ---------------------------------------- #


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def main() -> None:
    log("Loading CSV...")
    df = pd.read_csv(DATA_PATH)

    log(f"Total rows: {len(df)}")

    # Fill NaNs once
    for col in EMBED_FIELDS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = df[col].fillna("")

    # Build corpus (do NOT use only `uses`; include generic + composition)
    log("Building text corpus...")
    corpus = (
        df["uses"] + " " +
        df["generic_name"] + " " +
        df["composition"]
    ).tolist()

    # Load model
    log("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    all_embeddings = []
    total = len(corpus)

    log("Starting embedding generation (this is slow on CPU - normal)...")

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk = corpus[start:end]

        log(f"Embedding rows {start} -> {end}")

        emb = model.encode(
            chunk,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")

    log(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Build FAISS index (Inner Product)
    log("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    log(f"FAISS index size: {index.ntotal}, dim: {index.d}")

    # Save index
    log("Saving FAISS index...")
    faiss.write_index(index, str(INDEX_DIR / "medicine.index"))

    # Save metadata (full dataframe)
    log("Saving metadata...")
    df.to_pickle(INDEX_DIR / "medicine_metadata.pkl")

    log("INDEX BUILD COMPLETE")


if __name__ == "__main__":
    main()
