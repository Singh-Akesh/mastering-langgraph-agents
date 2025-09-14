from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

COLLECTION_NAME = "emails"
VECTOR_SIZE = 768  # dimension of your embeddings
DISTANCE = "Cosine"  # or "Dot", "Euclid"

client = QdrantClient(url="http://localhost:6333")


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
)

def query_qdrant():
    search_result = client.query_points(
        collection_name="star_charts",
        query=[0.2, 0.1, 0.9, 0.7],
        with_payload=True,
        limit=3
    ).points
    print(search_result)
