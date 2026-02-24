import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
You are a helpful assistant.

Answer ONLY from the context below.
If the answer is not present, say "I don't know."

Context:
{context_text}

Question: {query}

Answer:
"""


def generate_answer(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
    )

    return response.json()["response"]