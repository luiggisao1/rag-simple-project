# Simple RAG Test

A minimal, lightweight test project to experiment with Retrieval-Augmented Generation (RAG) using local LLMs via Ollama.

## Overview

This project tests a basic RAG pipeline from scratch:
1. **Document Loading & Chunking**: Reads markdown files from `docs/` and splits them into paragraphs.
2. **Embeddings**: Generates vector embeddings for each chunk using Ollama (`embeddinggemma`).
3. **Retrieval**: Uses cosine similarity (vector dot product normalized by norms) with NumPy to find top-`k` relevant chunks matching the user's query.
4. **Generation**: Prompts an LLM (`qwen3:4b-instruct`) via Ollama to answer questions strictly grounded in the retrieved context.

## Prerequisites

1. **Ollama**: Ensure [Ollama](https://ollama.com/) is installed and running (`http://localhost:11434`).
2. **Download Models**:
   ```bash
   ollama pull embeddinggemma
   ollama pull qwen3:4b-instruct
   ```
3. **Python & Dependencies**:
   Python 3.10+ (or Python 3.14 via `uv`) with `numpy` and `requests`.

   ```bash
   uv sync
   # or
   pip install requests numpy
   ```

## Usage

Run the RAG script:

```bash
python rag.py
# or with uv:
uv run rag.py
```

Type your question at the prompt (e.g., questions about the sample employee handbook in `docs/handbook.md`).
