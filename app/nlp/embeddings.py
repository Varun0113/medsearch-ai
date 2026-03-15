from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def encode(texts):
    return model.encode(texts, convert_to_numpy=True)
