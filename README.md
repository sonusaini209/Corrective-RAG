# Corrective RAG (CRAG)

Ask questions about a PDF/TXT file. Retrieved chunks are graded by an LLM; if the document
can't answer the question, the system falls back to web search before answering.
Based on [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884).

## How it works

```
question -> retrieve (FAISS, top 4) -> grade each chunk (0-1)
   CORRECT   (any chunk > 0.7)        -> refine -> generate
   AMBIGUOUS (mixed scores)           -> rewrite query -> Tavily web search -> refine (doc + web) -> generate
   INCORRECT (all chunks < 0.3)       -> rewrite query -> Tavily web search -> refine (web only)   -> generate
```

- **Refine**: context is split into sentences, de-duplicated, and each sentence is kept or dropped by an LLM filter.
- **Stack**: FastAPI, LangGraph, FAISS, MiniLM embeddings, Groq (`openai/gpt-oss-120b`), Tavily.
- The API returns the verdict, chunk scores, rewritten web query and sources, so you can see why CRAG chose its path.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and TAVILY_API_KEY
uvicorn main:app --reload
```

Open http://127.0.0.1:8000, upload a file, ask a question.

## API

- `POST /upload` (multipart `file`, .pdf or .txt, max 10 MB) -> `{doc_id, chunks}`
- `POST /ask` `{question, doc_id}` -> `{answer, verdict, reason, scores, web_query, doc_sources, web_sources}`
- `GET /health`

## Limitations

- Indexes live in memory (last 20 documents), so they are lost on restart and don't work across multiple workers.
- Scanned PDFs (no text layer) are not supported.
- Single-turn Q&A; no chat history.
