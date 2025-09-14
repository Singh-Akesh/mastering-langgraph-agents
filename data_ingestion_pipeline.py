import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import PointStruct

from qdrant_client import QdrantClient
from email_util import fetch_emails, extract_email_content

# ---------- CONFIG ----------
BATCH_SIZE = 1 # number of emails to fetch per run
UID_FILE = "last_uid.txt"
QDRANT_URL = "http://localhost:6333"   # or Qdrant Cloud endpoint
COLLECTION_NAME = "emails"
VECTOR_SIZE = 768  # dimension of your embeddings
DISTANCE = "Cosine"  # or "Dot", "Euclid"
# ----------------------------

# Initialize Ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url=QDRANT_URL)


def load_last_uid():
    if os.path.exists(UID_FILE):
        with open(UID_FILE, "r") as f:
            return int(f.read().strip())
    return None

def save_last_uid(uid):
    with open(UID_FILE, "w") as f:
        f.write(str(uid))


def chunk_and_embed(docs):
    """Split documents into chunks and create embeddings"""
    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.create_documents([docs])

    # Create embeddings for each chunk
    # vectors = embeddings.embed_documents([d.page_content for d in split_docs])
    texts = [d.page_content for d in split_docs]
    vectors = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        vectors.extend(embeddings.embed_documents(batch))

        print(f"{len(vectors)} processed out of {len(texts)}")

    # Return both text chunks and their embeddings (parallel lists)
    return split_docs, vectors

def ingest_into_qdrant(docs, embeddings, uid):
    """Push chunks into Qdrant with Seq metadata"""

    for doc in docs:
        doc.metadata["email_seq"] = uid

    points = [
        PointStruct(
            id=uid * 1000 + idx,  # uid = email UID, idx = chunk index
            vector=v,
            payload={"text": d.page_content, "email_uid": uid}
        )
        for idx, (d, v) in enumerate(zip(docs, embeddings))
    ]

    # Upsert into collection
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(docs)} chunks from email Sequence Id {uid}")


def main():
    last_uid = load_last_uid()
    emails = fetch_emails(limit=BATCH_SIZE, since_uid=last_uid)

    for seq, msg in emails:
        content = extract_email_content(msg)
        if not content.strip():
            continue
        docs, embeddings = chunk_and_embed(content)
        ingest_into_qdrant(docs, embeddings, uid=seq)
        save_last_uid(seq)  # update after processing


if __name__ == "__main__":
    main()
