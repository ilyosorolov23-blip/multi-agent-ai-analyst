"""
F2 - Ingestion & vector store. Loads documents, chunks them, embeds them
with Gemini embeddings, and stores them in Qdrant. Uses *embedded* Qdrant
(a local on-disk collection, no server, no signup) unless QDRANT_URL is
set, in which case it talks to a free cloud.qdrant.io cluster instead.
"""
import glob
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.config import settings
from app.llm import get_embeddings

# gemini-embedding-001 (served by the proxy as "gemini-embedding") defaults
# to 3072-dim output, NOT the 768 of the old text-embedding-004. If you had
# an existing Qdrant collection built with the old model, its vectors are
# now the wrong size — either delete/recreate the collection or point
# QDRANT_COLLECTION at a fresh name before re-ingesting.
EMBED_DIM = 3072

_embeddings = get_embeddings()


def _client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    # embedded mode: persists to disk, needs no running server / signup
    return QdrantClient(path="./data/qdrant_local")


def get_vectorstore(collection: str | None = None) -> QdrantVectorStore:
    collection = collection or settings.qdrant_collection
    client = _client()
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=collection, embedding=_embeddings)


def ingest_folder(folder: str = "./data/docs", collection: str | None = None) -> int:
    """Loads every .txt/.md file in `folder`, chunks it, embeds it, stores it.
    Returns the number of chunks written. This is F2's 'Done when' check."""
    paths = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
    paths = [p for p in paths if p.lower().endswith((".txt", ".md"))]
    if not paths:
        raise FileNotFoundError(f"No .txt/.md files found under {folder}")

    docs = []
    for p in paths:
        docs.extend(TextLoader(p, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    store = get_vectorstore(collection)
    store.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    n = ingest_folder()
    print(f"Ingested {n} chunks into Qdrant collection '{settings.qdrant_collection}'.")
