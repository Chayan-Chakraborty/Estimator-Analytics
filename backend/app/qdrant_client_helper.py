from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "items"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Compatibility shim for older qdrant-client versions.
# Some versions expose `search_points()` instead of `search()`.
if not hasattr(client, "search") and hasattr(client, "search_points"):
    _search_points = getattr(client, "search_points")

    def _search(*args, **kwargs):
        return _search_points(*args, **kwargs)

    setattr(client, "search", _search)

def create_collection(vector_size=384):
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )