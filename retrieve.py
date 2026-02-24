import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_store(index_path="faiss.index"):
    index = faiss.read_index(index_path)

    with open("metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve(query, index, chunks, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    results = [chunks[i]["text"] for i in indices[0]]
    return results