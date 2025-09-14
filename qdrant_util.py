from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

COLLECTION_NAME = "emails"
VECTOR_SIZE = 768  # dimension of your embeddings
DISTANCE = "Cosine"  # or "Dot", "Euclid"

client = QdrantClient(url="http://localhost:6333")

def create_collection():
    # Create collection if it doesn't exist
    if COLLECTION_NAME in [col.name for col in client.get_collections().collections]:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return
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

def count_emails():
    # Get exact number of points
    count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
    exact_point_count = count_result.count
    print(f"Exact number of points: {exact_point_count}")

if __name__ == '__main__':
    count_emails()

