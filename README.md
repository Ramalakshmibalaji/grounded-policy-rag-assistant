# The Grounded Answer

Date-aware RAG assistant for the Calder County Household Support Program.

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads `all-MiniLM-L6-v2` through sentence-transformers.

## Current corpus

The `data/` folder currently contains Amendment No. 2026-01 supplied for the Brite Spark Day 2 challenge.

**Important:** add the original consolidated policy manual supplied for Problem 1 to `data/` before final submission. The amendment modifies that manual; it does not replace it.

## Design

- Sentence-transformer embeddings for semantic retrieval.
- BM25-style lexical retrieval.
- 72/28 hybrid ranking.
- Date-aware amendment handling.
- Exact clause references.
- Conservative "I don't know" fallback.
- No paid services required for the core demo.

## Clean-environment check

Run the exact README commands from a fresh environment before submission.
