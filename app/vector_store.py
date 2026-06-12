"""
Vector store wrapper around ChromaDB.
Uses ChromaDB's built-in ONNX embedding function — no torch, no triton.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_documents"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.45  # chunks below this are too weak — filter them out


class VectorStore:
    def __init__(self):
        self.embedding_fn = ONNXMiniLM_L6_V2()
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [
            f"{doc.metadata.get('source', 'unknown')}__chunk_{i}"
            for i, doc in enumerate(documents)
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        return len(documents)

    def similarity_search(self, query: str, k: int = TOP_K) -> List[Document]:
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = round(1 - distance, 4)

            # Drop chunks below threshold before they even reach the grader
            # This is the key fix — weak matches (0.33-0.39) never get through
            if score < SIMILARITY_THRESHOLD:
                print(f"  [filtered] score={score} below threshold — skipping")
                continue

            doc = Document(
                page_content=text,
                metadata={**metadata, "similarity_score": score}
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


vector_store = VectorStore()
