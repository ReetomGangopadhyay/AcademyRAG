from typing import List, Dict, Any
from .store import get_store
from .embed import Embedder

def retrieve_with_rerank(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    store = get_store()
    embedder = Embedder()
    q_vec = embedder.embed([query])[0]
    res = store.query(query_embeddings=[q_vec], n_results=top_k, include=["documents","metadatas","distances"])
    docs = []
    if res and res.get("documents"):
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            item = {"text": doc, "metadata": meta, "score": 1.0/(1.0+dist) if dist is not None else None}
            docs.append(item)
    return docs
