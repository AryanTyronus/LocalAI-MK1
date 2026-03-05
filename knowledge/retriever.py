"""
Local retrieval pipeline:
query embedding -> vector search -> top chunk results.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

from core.logger import logger
from knowledge.document_loader import KnowledgeDocumentLoader
from knowledge.embeddings import EmbeddingService
from knowledge.vector_store import KnowledgeVectorStore


class KnowledgeRetriever:
    def __init__(self):
        self._loader = KnowledgeDocumentLoader()
        self._embeddings = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
        self._store = KnowledgeVectorStore()
        self._ready = False
        self._documents: List[Dict] = []

    def ensure_ready(self) -> None:
        if self._ready:
            return
        if self._store.load():
            chunks = self._store.list_chunks()
            self._documents = self._build_documents(chunks)
            self._ready = True
            logger.info(f"KnowledgeRetriever loaded persisted index ({len(chunks)} chunks)")
            return

        chunks = self._loader.load_chunks()
        if not chunks:
            self._documents = []
            self._ready = True
            logger.info("KnowledgeRetriever loaded no chunks")
            return
        vectors = self._embeddings.encode([c.get("text", "") for c in chunks])
        self._store.build(chunks, vectors)
        self._documents = self._build_documents(chunks)
        self._ready = True
        logger.info(f"KnowledgeRetriever built index ({len(chunks)} chunks)")

    def _build_documents(self, chunks: List[Dict]) -> List[Dict]:
        docs: "OrderedDict[str, Dict]" = OrderedDict()
        for row in chunks:
            doc_id = row["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": row.get("doc_name", ""),
                    "path": row.get("path", ""),
                    "source_type": row.get("source_type", "unknown"),
                    "num_pages": 0,
                }
            docs[doc_id]["num_pages"] = max(
                int(docs[doc_id]["num_pages"]),
                int(row.get("page_number", 1)),
            )
        return list(docs.values())

    def list_documents(self) -> List[Dict]:
        self.ensure_ready()
        return list(self._documents)

    def list_chunks(self) -> List[Dict]:
        self.ensure_ready()
        return self._store.list_chunks()

    def search(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict]:
        self.ensure_ready()
        if not query.strip():
            return []
        query_vec = self._embeddings.encode_one(query)
        if query_vec.size == 0:
            return []
        return self._store.search(query_vec=query_vec, top_k=top_k, doc_id=doc_id)

