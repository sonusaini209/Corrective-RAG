import re
from typing import List, TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config import llm  # your existing config.py (defines `llm`)

load_dotenv()

# ---------- Index (built once at startup) ----------
docs = PyPDFLoader("the-metamorphosis.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 4})

UPPER_TH, LOWER_TH = 0.7, 0.3


class State(TypedDict, total=False):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str


# ---------- Retrieve ----------
def retrieve_node(state: State):
    return {"docs": retriever.invoke(state["question"])}


# ---------- Evaluate each chunk ----------
class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for RAG.\n"
     "You will be given ONE retrieved chunk and a question.\n"
     "Return a relevance score in [0.0, 1.0].\n"
     "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Be conservative with high scores.\n"
     "Also return a short reason.\nOutput JSON only."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
]) | llm.with_structured_output(DocEvalScore, method="json_mode")


def eval_each_doc_node(state: State):
    scores, good = [], []
    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": state["question"], "chunk": d.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {UPPER_TH}."}
    if scores and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {LOWER_TH}."}
    return {"good_docs": good, "verdict": "AMBIGUOUS",
            "reason": f"No chunk > {UPPER_TH}, but not all < {LOWER_TH}."}


# ---------- Refine (sentence-level filter) ----------
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool


filter_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance filter.\n"
     "Return keep=true only if the sentence directly helps answer the question.\n"
     "Use ONLY the sentence. Output JSON only."),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
]) | llm.with_structured_output(KeepOrDrop, method="json_mode")


def refine(state: State):
    v = state.get("verdict")
    if v == "CORRECT":
        use = state["good_docs"]
    elif v == "INCORRECT":
        use = state["web_docs"]
    else:
        use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in use).strip()
    kept = [s for s in decompose_to_sentences(context)
            if filter_chain.invoke({"question": state["question"], "sentence": s}).keep]
    return {"kept_strips": kept, "refined_context": "\n".join(kept).strip()}


# ---------- Rewrite + Web search ----------
class WebQuery(BaseModel):
    query: str


rewrite_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n- Keep it short (6–14 words).\n"
     "- If the question implies recency (recent/latest/last week/last month), add a constraint like (last 30 days).\n"
     "- Do NOT answer the question.\n- Return JSON with a single key: query"),
    ("human", "Question: {question}"),
]) | llm.with_structured_output(WebQuery, method="json_mode")


def rewrite_query_node(state: State):
    return {"web_query": rewrite_chain.invoke({"question": state["question"]}).query}


tavily = TavilySearchResults(max_results=5)


def web_search_node(state: State):
    results = tavily.invoke({"query": state.get("web_query") or state["question"]})
    web_docs = []
    for r in results or []:
        url, title = r.get("url", ""), r.get("title", "")
        content = r.get("content", "") or r.get("snippet", "")
        web_docs.append(Document(
            page_content=f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}",
            metadata={"url": url, "title": title}))
    return {"web_docs": web_docs}


# ---------- Generate ----------
answer_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
     "If the context is empty or insufficient, say: 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}"),
]) | llm


def generate(state: State):
    return {"answer": answer_chain.invoke(
        {"question": state["question"], "context": state["refined_context"]}).content}


# ---------- Graph ----------
def route_after_eval(state: State) -> str:
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"


g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges("eval_each_doc", route_after_eval,
                        {"refine": "refine", "rewrite_query": "rewrite_query"})
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)
crag = g.compile()


# ---------- API ----------
app = FastAPI(title="CRAG API")


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):  # sync def -> runs in threadpool, doesn't block server
    res = crag.invoke({"question": body.question, "docs": [], "good_docs": [],
                       "web_docs": [], "kept_strips": []})
    return {
        "answer": res["answer"],
        "verdict": res["verdict"],
        "reason": res["reason"],
        "web_query": res.get("web_query", ""),
        "sources": [d.metadata.get("url") for d in res["web_docs"]] if res.get("web_docs") else [],
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
