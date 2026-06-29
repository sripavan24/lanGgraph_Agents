import os
import hashlib
import math
from getpass import getpass
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

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

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        value = getpass(f"Enter {name}: ").strip()
    if not value:
        raise RuntimeError(f"{name} is required. Add it to .env or enter it when prompted.")
    return value

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=require_env("OPENAI_API_KEY")
)

embeddings = LocalHashEmbeddings()
