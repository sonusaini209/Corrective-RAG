from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# define your LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,              
    model="openai/gpt-oss-120b",   
    temperature=0.7,
)
