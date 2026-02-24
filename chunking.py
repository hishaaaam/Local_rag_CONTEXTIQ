def chunk_text(text, chunk_size=350, overlap=50):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_documents(documents):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"]
            })

    return all_chunks