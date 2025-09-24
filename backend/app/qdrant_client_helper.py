from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "items"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection(vector_size=384):
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )