import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Generation LLM: a little creativity is fine for the final answer.
llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0.3)

# Judge LLM: chunk grader, sentence filter and query rewriter must be deterministic.
judge_llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0)
