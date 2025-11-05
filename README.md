# AcademyRAG - Learning Assistant

AcademyRAG is a Retrieval-Augmented Generation (RAG) based learning assistant that helps with learning by providing access to your specific knowledge base.

## Features

- Smart document search with semantic understanding
- Support for multiple document formats (PDF, PPTX, MD, TXT)
- Interactive Q&A with source citations
- Guided topic summaries via "Teach Me" mode
- Automatic quiz generation for learning assessment
- Works completely locally with option for OpenAI integration

## Quick Start

```bash
# 1) Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2) Install the package
pip install -e .

# 3) Configure environment
# Create .env file with:
ACADEMYRAG_EMBED_PROVIDER=sentence-transformers
ACADEMYRAG_ST_MODEL=all-MiniLM-L6-v2
ACADEMYRAG_LLM_PROVIDER=none  # or 'openai' if using OpenAI
ACADEMYRAG_DB_DIR=./data/index
ACADEMYRAG_CHUNK_SIZE=900
ACADEMYRAG_CHUNK_OVERLAP=120

# 4) Run the application
streamlit run app/main.py
```

## Project Structure

```text
academyrag/
├── app/
│   ├── __init__.py
│   ├── components.py
│   └── main.py          # Streamlit interface
├── rag/
│   ├── chunk.py         # Text chunking logic
│   ├── embed.py         # Embedding generation
│   ├── eval.py          # Evaluation utilities
│   ├── generate.py      # Text generation
│   ├── ingest.py        # Document ingestion
│   ├── retrieve.py      # Document retrieval
│   └── store.py         # Vector store management
├── data/
│   ├── index/          # ChromaDB storage
│   └── raw/            # Document storage
└── setup.py            # Package configuration
```

## Usage Modes

### 1. No-LLM Mode (Default)

- Uses only embeddings and retrieval
- Shows relevant document snippets
- No generation capabilities
- Completely local operation

### 2. Full Mode (with OpenAI)

- Enables answer generation
- Provides summarization
- Creates interactive quizzes
- Requires API key

## Example Queries

- "What are common cost drivers in a value chain? Cite sources with page numbers."
- "Teach me the basics of market entry analysis."
- "Quiz me on cost drivers (5 MCQs)."

## Technical Details

- **Document Processing**:
  - Supports PDF, PPTX, MD, and TXT formats
  - Uses `pymupdf` for PDF extraction
  - Uses `python-pptx` for PowerPoint files

- **Embeddings**:
  - Default: Sentence Transformers (`all-MiniLM-L6-v2`)
  - Optional: OpenAI (`text-embedding-3-large`)

- **Vector Store**:
  - Uses ChromaDB
  - Persistent storage in `./data/index`
  - Efficient similarity search

- **Text Chunking**:
  - Default chunk size: 900 tokens
  - Overlap: 120 tokens
  - Ensures context coherence

## Safety Features

- Answers are strictly based on retrieved context
- All responses include source citations
- Clear indication when no relevant context is found
- Suggestion system for improving document coverage

## Requirements

- Python 3.12 or higher
- Dependencies listed in `requirements.txt`
- Optional: OpenAI API key for enhanced features
