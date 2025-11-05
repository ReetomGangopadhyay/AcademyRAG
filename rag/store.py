import os
import chromadb
from chromadb.utils import embedding_functions

_client = None
_collection = None

def get_store():
    global _client, _collection
    if _client is None:
        db_dir = os.getenv("ACADEMYRAG_DB_DIR", "./data/index")
        os.makedirs(db_dir, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_dir)
    if _collection is None:
        _collection = _client.get_or_create_collection("academyrag")
    return _collection
