"""
Professional Local RAG App — Streamlit Edition
------------------------------------------------
Features:
- Modern professional layout
- Sidebar document processing
- Chat-style interface
- Cached models for speed
- Fully local (FAISS + SentenceTransformers + Ollama)

Run:
    streamlit run rag_gradio_ui.py
"""

import os
import numpy as np
import faiss
import pdfplumber
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
st.write("App started")
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Local RAG Pro",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# CONFIG
# ============================================================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

# ============================================================
# CACHED MODEL LOADERS
# ============================================================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL_NAME)

with st.spinner("🚀 Loading AI engine..."):
    embed_model = load_embed_model()

# ❗ DO NOT load reranker at startup
reranker = None

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks_metadata = None
    st.session_state.chat_history = []

# ============================================================
# HELPERS
# ============================================================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=350, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_index_from_files(files):
    all_chunks = []

    for uploaded_file in files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)

            for ch in chunks:
                all_chunks.append({
                    "text": ch,
                    "source": uploaded_file.name,
                })

    if not all_chunks:
        return False, "❌ No extractable text found."

    texts = [c["text"] for c in all_chunks]

    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))

    st.session_state.index = index
    st.session_state.chunks_metadata = all_chunks

    return True, f"✅ Indexed {len(all_chunks)} chunks from {len(files)} file(s)."


def retrieve(query, top_k=2):
    index = st.session_state.index
    metadata = st.session_state.chunks_metadata

    if index is None:
        return []

    q_emb = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    return [metadata[i]["text"] for i in indices[0]]


def rerank_docs(query, docs, top_k=2):
    global reranker

    if not docs:
        return []

    # 🔥 Lazy load (only first time)
    if reranker is None:
        with st.spinner("⚙️ Loading reranker..."):
            reranker = load_reranker()

    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)

    ranked = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return ranked[:top_k]


def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
You are a helpful assistant.
Answer ONLY from the context below.
If the answer is not present, say \"I don't know.\"

Context:
{context_text}

Question: {query}

Answer:
"""


def generate_answer(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 200, "temperature": 0},
        },
        timeout=300,
    )

    return response.json().get("response", "❌ LLM error")


# ============================================================
# SIDEBAR — DOCUMENT INGESTION
# ============================================================
st.sidebar.title("📂 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.sidebar.button("⚡ Build Index", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one PDF.")
    else:
        success, msg = build_index_from_files(uploaded_files)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Fully local • No API costs")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("🧠 ContextIQ")
st.caption("Chat with your documents — fast, private, and professional.")

# ============================================================
# CHAT UI
# ============================================================
chat_container = st.container()

with chat_container:
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

user_query = st.chat_input("Ask something about your documents...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    with st.chat_message("assistant"):
        if st.session_state.index is None:
            response = "⚠️ Please upload and index documents first."
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                retrieved = retrieve(user_query, top_k=3)
                reranked = rerank_docs(user_query, retrieved, top_k=2)
                prompt = build_prompt(user_query, reranked)
                response = generate_answer(prompt)
                st.markdown(response)

    st.session_state.chat_history.append(("assistant", response))
