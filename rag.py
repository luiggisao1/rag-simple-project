import requests
import numpy as np
from pathlib import Path


OLLAMA = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "qwen3:4b-instruct"

def embed(texts: list[str]) -> np.ndarray:
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    return np.array(r.json()["embeddings"])


def ask_llm(prompt: str) -> str:
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    })
    return r.json()["message"]["content"]

chunks = []
for file in Path("docs").glob("*.md"):
    chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]

chunk_vectors = embed(chunks)

def retrieve(question: str, k=3):
    q = embed([question])[0]
    scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]


while True:
    question = input("Ask: ")
    context = "\n\n".join(retrieve(question))
    prompt = (
        "Answer the question using ONLY the context below."
        "If the answer isn't in the context, say 'I don't know'."
        f"\n\nContext:\n{context}"
        f"\n\nQuestion:\n{question}"
    )
    print(ask_llm(prompt))
