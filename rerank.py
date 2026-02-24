from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, docs, top_k=2):
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(zip(scores, docs), reverse=True)
    ]

    return ranked_docs[:top_k]