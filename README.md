# RAG Pipeline: Multi-Query Expansion + Reranking + Eval

A RAG system over a Microsoft annual report PDF, built to compare three
retrieval strategies and measure whether each one actually improves results
— not just assumed to.

## Architecture

| Module | Responsibility |
|---|---|
| `document_store.py` | PDF parsing, chunking, vector DB indexing + bi-encoder similarity search |
| `llm.py` | `LLMClient` (Groq API wrapper), `QueryExpander` (multi-query generation), `Generator` (final answer generation + synthetic question generation for eval) |
| `reranker.py` | Second-stage reranking over a candidate pool |
| `pipeline.py` | `RAGPipeline` — composes the above into three retrieval strategies: `naive`, `multiquery`, `reranked` |
| `evaluator.py` | Builds a synthetic eval set to check accuracy of various rag query strategies |
| `main.py` | Wires everything together: runs the eval comparison, then an interactive query loop |


## Key result (from an n=15 synthetic eval set)

| Strategy | Recall |
|---|---|
| naive | 0.87 |
| multiquery | 0.93 |
| reranked | 0.93 |

Multi-query expansion measurably improved recall over naive single-query
retrieval. Reranking held recall steady on this run — sample size might've been too small

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # then fill in your GROQ_API_KEY
```

Place the source PDF at `pdfs/microsoft-annual-report.pdf` (or update
`config.PDF_PATH`).

## Run

```bash
python main.py
```

Runs the strategy comparison first, then drops into an interactive prompt
using multi-query + reranking.

# project-rag
