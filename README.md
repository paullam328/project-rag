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

Place the source PDF `microsoft-annual-report.pdf` at `pdfs/microsoft-annual-report.pdf`, in the same directory as where you pulled `project-rag` (outside of this project's directory). <-- Important. Need that microsoft-annual-report.pdf to scrape data from.

If you want to scrape with manual pdfs, you can simply:
1. Place your pdf file in the same directory as where you pulled `project-rag` (outside of this project's directory) as `pdfs/<file-name>.pdf`
2. Update `config.PDF_PATH` to `pdfs/<file-name>.pdf`
3. Enjoy!

## Gettng Groq API key (Need this to talk to LLM)
1. visit https://groq.com/
2. Developer tab --> Free API key
3. Create .env file within ./project-rag
4. Within .env file: GROQ_API_KEY=`<API-Key>`

## Run

```bash
# Install dependencies using requirement.txt and venv:
python3 -m venv .venv # outside of project-rag repo, setup virtual env
source .venv/bin/activate
cd ./project-rag
pip3 install -r requirements.txt

# Compile:
cd .. # Step back a directory level outside of project-rag
python3 -m project-rag
```

Runs the strategy comparison first, then drops into an interactive prompt
using multi-query + reranking.

# project-rag
