import os
import hashlib
import math
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values, find_dotenv, load_dotenv

DOTENV_PATH = find_dotenv(usecwd=True)

def load_project_env() -> None:
    if not DOTENV_PATH:
        return
    load_dotenv(DOTENV_PATH, encoding="utf-8-sig")
    for key, value in dotenv_values(DOTENV_PATH, encoding="utf-8-sig").items():
        normalized_key = (key or "").lstrip("\ufeff").strip()
        if normalized_key and value and not os.environ.get(normalized_key):
            os.environ[normalized_key] = value
    if not os.environ.get("GROQ_API_KEY") and os.environ.get("GROK_API_KEY"):
        os.environ["GROQ_API_KEY"] = os.environ["GROK_API_KEY"]

load_project_env()

class LocalHashEmbeddings(Embeddings):
    """Small local embedding fallback that does not require an API key."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

def require_env(name: str, required: bool = False) -> str | None:
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"{name} is required. Add {name}=your_key to .env.")
    return value

GROQ_API_KEY = require_env("GROQ_API_KEY") or require_env("GROK_API_KEY")

if GROQ_API_KEY:
    llm = ChatOpenAI(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=GROQ_API_KEY,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0.2,
    )
else:
    llm = None

embeddings = LocalHashEmbeddings()
