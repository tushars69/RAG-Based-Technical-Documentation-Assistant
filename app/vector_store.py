"""
Vector store wrapper around ChromaDB.

Handles:
- Embedding documents using sentence-transformers (free, local)
- Storing chunks with metadata
- Similarity search at query time
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, 384-dim, no API key needed
TOP_K = 5


class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embed(texts)

        # Stable IDs so re-running ingest never creates duplicates
        ids = [
            f"{doc.metadata.get('source', 'unknown')}__chunk_{i}"
            for i, doc in enumerate(documents)
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(documents)

    def similarity_search(self, query: str, k: int = TOP_K) -> List[Document]:
        query_embedding = self.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            doc = Document(
                page_content=text,
                metadata={**metadata, "similarity_score": round(1 - distance, 4)}
            )
            documents.append(doc)

        return documents

    def list_documents(self) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        seen = set()
        docs = []

        for metadata in results["metadatas"]:
            source = metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                docs.append({
                    "source": source,
                    "title": metadata.get("title", source),
                })
        return docs

    def get_chunk_count(self) -> int:
        return self.collection.count()


# Singleton shared across the entire app
vector_store = VectorStore()
