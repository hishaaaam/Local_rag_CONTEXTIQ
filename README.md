# 🧠 ContextIQ — Local RAG Assistant

A professional, fully local **Retrieval-Augmented Generation (RAG)** system built from scratch in Python — no LangChain, no paid APIs.

Upload PDFs and chat with your documents using **FAISS + SentenceTransformers + Ollama (phi3:mini)** — fast, private, and free.

---

## ✨ Features

* 📂 Upload and process multiple PDFs
* 🔍 Semantic search with FAISS
* 🧠 Local embeddings (MiniLM)
* 🎯 Cross-encoder reranking for better accuracy
* 💬 Modern Streamlit chat interface
* 🔒 100% local — no API costs
* ⚡ Cached models for faster reloads
* 🧩 Clean modular pipeline

---

## 🏗️ System Architecture

```text
Documents → Chunking → Embeddings → FAISS → Retrieval → Reranking → Local LLM → Answer
```

### How it works

1. **Ingestion** — Extract text from PDFs
2. **Chunking** — Split into semantic chunks
3. **Embedding** — Convert text into vectors
4. **Vector Search** — Retrieve relevant chunks
5. **Reranking** — Improve relevance
6. **Generation** — Local LLM produces grounded answer

---

## 🧰 Tech Stack

* **Python**
* **Streamlit**
* **Sentence Transformers**
* **FAISS**
* **Ollama (phi3:mini)**
* **pdfplumber**
* **PyTorch**

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows
# source .venv/bin/activate  # Mac/Linux
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have it yet:

```bash
pip install streamlit pdfplumber sentence-transformers faiss-cpu numpy requests torch
```

---

### 4️⃣ Install Ollama (Required)

Download and install from:

👉 https://ollama.com/download

Pull the local model:

```bash
ollama pull phi3:mini
```

Verify installation:

```bash
ollama list
```

---

### 5️⃣ Run the application

```bash
streamlit run rag_gradio_ui.py
```

Open the local URL shown in the terminal.

---

## 📖 Usage

1. Upload PDFs from the sidebar
2. Click **Build Index**
3. Ask questions in the chat
4. Get answers grounded in your documents

---

## ⚡ Performance Notes

* First startup is slower due to model loading
* Subsequent runs are faster (cached)
* Works fully offline after models are downloaded
* CPU-only systems may have slower inference

---

## 🚧 Current Limitations

* Scanned/image PDFs are not supported
* Very large document sets may slow indexing
* Hybrid search not implemented yet

---

## 🔮 Future Improvements

* Hybrid search (BM25 + vector)
* Streaming responses
* Source citations panel
* Dark/light theme toggle
* Docker deployment
* Multi-user support

---

## 📁 Recommended Project Structure

```
rag-project/
│
├── rag_gradio_ui.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/ (optional)
```

---

## 👨‍💻 Author

**Hisham Hidayathulla**

* GitHub: https://github.com/hishaaaam
* LinkedIn: https://www.linkedin.com/in/hisham-hidaya/

---

## ⭐ Support

If you found this useful, consider starring the repository!

---
