import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_store(chunks, index_path="faiss.index"):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, index_path)

    # Save metadata
    with open("metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return index