import os
from typing import List
from pydantic import BaseModel
from tqdm import tqdm

# Providers broke af so no openai lol
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class Embedder(BaseModel):
    provider: str = "openai"  # or "sentence-transformers"
    model: str = "text-embedding-3-large"
    st_model: str = "all-MiniLM-L6-v2"

    def _get_openai(self):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        provider = os.getenv("ACADEMYRAG_EMBED_PROVIDER", self.provider)
        if provider == "sentence-transformers":
            model_name = os.getenv("ACADEMYRAG_ST_MODEL", self.st_model)
            st_model = SentenceTransformer(model_name)
            vecs = st_model.encode(texts, normalize_embeddings=True).tolist()
            return vecs
        else:
            client = self._get_openai()
            resp = client.embeddings.create(model=os.getenv("ACADEMYRAG_EMBED_MODEL", self.model),
                                            input=texts)
            return [d.embedding for d in resp.data]
