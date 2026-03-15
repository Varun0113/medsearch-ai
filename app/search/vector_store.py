import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings: np.ndarray):
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query_vec, top_k=3):
        query_vec = query_vec.astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        return indices[0], distances[0]
