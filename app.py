from ingestion import load_documents
from chunking import process_documents
from embed_store import build_vector_store
from retrieve import load_store, retrieve
from rerank import rerank
from generate import build_prompt, generate_answer

DATA_PATH = "data"


def setup():
    docs = load_documents(DATA_PATH)
    chunks = process_documents(docs)
    build_vector_store(chunks)


def ask(query):
    index, chunks = load_store()

    retrieved = retrieve(query, index, chunks)
    reranked = rerank(query, retrieved)

    prompt = build_prompt(query, reranked)
    answer = generate_answer(prompt)

    return answer


if __name__ == "__main__":
    # Run once to build index
    # setup()

    while True:
        q = input("\nAsk: ")
        print("\nAnswer:", ask(q))