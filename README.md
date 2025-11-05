# AcademyRAG — Onboarding & Upskilling Assistant (MVP)

**Goal:** Ask questions about your training docs and get concise answers with citations.
Includes “Teach Me” (guided summaries) and “Quiz Me” (5 MCQs) modes.

## Quick Start

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Set your LLM/embedding provider in .env
#    - For OpenAI, set:
#      OPENAI_API_KEY=sk-...
#      ACADEMYRAG_EMBED_PROVIDER=openai        # default
#      ACADEMYRAG_LLM_PROVIDER=openai          # default
#    - For local embeddings (no key required), set:
#      ACADEMYRAG_EMBED_PROVIDER=sentence-transformers
#      ACADEMYRAG_ST_MODEL=all-MiniLM-L6-v2
#
# 4) Add documents (PDF, PPTX, MD, TXT) to ./data/raw
#
# 5) Ingest and run app
python -m rag.ingest --path data/raw
streamlit run app/main.py
```

## Project Layout
```
academyrag/
  app/
    main.py
    components.py
  rag/
    ingest.py
    chunk.py
    embed.py
    store.py
    retrieve.py
    generate.py
    prompts.py
    eval.py
  data/
    raw/         # put your PDFs/PPTX/MD/TXT here
    processed/   # extracted text/chunks
    index/       # vector store
  requirements.txt
  README.md
```

## Example Queries
- "What are common cost drivers in a value chain? Cite sources with page numbers."
- "Teach me the basics of market entry analysis."
- "Quiz me on cost drivers (5 MCQs)."

## Notes
- PDF/PPTX extraction relies on `pymupdf` and `python-pptx` (installed by default).
- Embeddings: OpenAI (`text-embedding-3-large`) by default; local fallback uses Sentence-Transformers (`all-MiniLM-L6-v2`).
- Vector store: ChromaDB (persisted under `./data/index`).

## Environment Vars (optional)
Create a `.env` at repo root:
```
OPENAI_API_KEY=...
ACADEMYRAG_EMBED_PROVIDER=openai            # or: sentence-transformers
ACADEMYRAG_LLM_PROVIDER=openai              # (OpenAI only in this MVP)
ACADEMYRAG_DB_DIR=./data/index
ACADEMYRAG_CHUNK_SIZE=900
ACADEMYRAG_CHUNK_OVERLAP=120
```

## Safety
- The app answers strictly from retrieved context and cites sources.
- If no relevant context is found, it will say so and suggest uploading more material.
