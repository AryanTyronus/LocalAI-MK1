"""
DocumentManager: handles PDF loading, token-based chunking, and embedding-based retrieval.

Features:
- Load multiple PDFs from the knowledge folder
- Assign unique document IDs
- Chunk pages into token-sized chunks (800 tokens, 100 overlap)
- Store metadata (document_id, doc_name, page_number, chunk_index)
- Build a FAISS index over chunk embeddings
- Support active document selection
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional
from pypdf import PdfReader
from core.config import KNOWLEDGE_FOLDER
from core.logger import logger
from core.model_manager import ModelManager


class DocumentManager:
    def __init__(self, chunk_tokens: int = 800, overlap: int = 100, index_file: Optional[str] = None):
        self.model_manager = ModelManager.get_instance()
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self.index_file = index_file

        # Loaded documents metadata
        self.documents: List[Dict] = []  # {doc_id, doc_name, path, num_pages}
        # Chunks list: each chunk is dict with id, doc_id, doc_name, page, chunk_index, text
        self.chunks: List[Dict] = []

        self.index = None
        self.embedding_dim = None

        self.active_doc_id: Optional[str] = None

        # Load existing docs
        self.load_documents()
        # Build index if chunks exist
        if self.chunks:
            self._build_index()

    # ---------------------- Document Loading ----------------------
    def load_documents(self):
        """Scan KNOWLEDGE_FOLDER for PDFs and load them as documents."""
        if not os.path.exists(KNOWLEDGE_FOLDER):
            logger.info("No knowledge folder found")
            return

        for filename in os.listdir(KNOWLEDGE_FOLDER):
            if not filename.lower().endswith('.pdf'):
                continue

            path = os.path.join(KNOWLEDGE_FOLDER, filename)
            doc_id = filename  # use filename as unique id
            logger.info(f"Loading document: {filename}")

            try:
                reader = PdfReader(path)
                num_pages = len(reader.pages)
                self.documents.append({'doc_id': doc_id, 'doc_name': filename, 'path': path, 'num_pages': num_pages})

                # Process pages into chunks
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    page_chunks = self._chunk_text(text)
                    for ci, chunk_text in enumerate(page_chunks):
                        chunk = {
                            'id': len(self.chunks),
                            'doc_id': doc_id,
                            'doc_name': filename,
                            'page_number': i,
                            'chunk_index': ci,
                            'text': chunk_text
                        }
                        self.chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to load PDF {filename}: {e}")

    # ---------------------- Chunking ----------------------
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using model tokenizer when available, otherwise fallback to whitespace tokens.
        Returns a list of token ids or tokens (strings) depending on tokenizer support.
        """
        tokenizer = getattr(self.model_manager, 'tokenizer', None)
        if tokenizer is not None:
            try:
                # Try common tokenizer APIs
                if hasattr(tokenizer, 'encode'):
                    return tokenizer.encode(text)
                if hasattr(tokenizer, '__call__'):
                    out = tokenizer(text)
                    if isinstance(out, dict) and 'input_ids' in out:
                        return out['input_ids']
            except Exception:
                logger.debug("Tokenizer.encode failed, falling back to whitespace tokenization")

        # Fallback: simple whitespace tokenization (tokens are words)
        return text.split()

    def _detokenize(self, tokens: List, use_ids: bool = True) -> str:
        """Convert tokens back to text. If tokenizer has decode, use it; else join whitespace tokens."""
        tokenizer = getattr(self.model_manager, 'tokenizer', None)
        if tokenizer is not None and use_ids:
            try:
                if hasattr(tokenizer, 'decode'):
                    return tokenizer.decode(tokens)
            except Exception:
                pass
        # Fallback
        return ' '.join(tokens)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into token-aware chunks with overlap and return list of chunk texts."""
        if not text:
            return []

        tokens = self._tokenize(text)
        use_ids = True
        if tokens and isinstance(tokens[0], str):
            use_ids = False

        chunks = []
        step = self.chunk_tokens - self.overlap
        if step <= 0:
            step = max(1, self.chunk_tokens // 2)

        # Slide window
        for start in range(0, len(tokens), step):
            end = start + self.chunk_tokens
            window = tokens[start:end]
            if not window:
                break

            if use_ids:
                try:
                    chunk_text = self._detokenize(window, use_ids=True)
                except Exception:
                    chunk_text = self._detokenize([str(t) for t in window], use_ids=False)
            else:
                chunk_text = self._detokenize(window, use_ids=False)

            chunks.append(chunk_text)

            if end >= len(tokens):
                break

        return chunks

    # ---------------------- Indexing ----------------------
    def _build_index(self):
        """Build a FAISS index over chunk embeddings."""
        if not self.chunks:
            self.index = None
            return

        texts = [c['text'] for c in self.chunks]
        embeddings = np.array(self.model_manager.embed(texts), dtype=np.float32)

        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)

        logger.info(f"Built document index with {len(self.chunks)} chunks (dim={self.embedding_dim})")

    def _ensure_index(self):
        if self.index is None:
            self._build_index()

    # ---------------------- Retrieval ----------------------
    def search(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict]:
        """Search for top_k most relevant chunks for the query. If doc_id provided, only search that document."""
        if not self.chunks:
            return []

        self._ensure_index()

        # If restricting to doc_id, build a temporary index on those chunks
        if doc_id:
            indices = [i for i, c in enumerate(self.chunks) if c['doc_id'] == doc_id]
            if not indices:
                return []

            embs = np.array([self.model_manager.embed([self.chunks[i]['text']])[0] for i in indices], dtype=np.float32)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(embs)

            q_emb = np.array(self.model_manager.embed([query]), dtype=np.float32)
            k = min(top_k, embs.shape[0])
            distances, I = idx.search(q_emb, k=k)

            results = []
            for dist, ii in zip(distances[0], I[0]):
                chunk_idx = indices[ii]
                results.append({'chunk': self.chunks[chunk_idx], 'score': float(dist)})

            return results

        # Global search
        q_emb = np.array(self.model_manager.embed([query]), dtype=np.float32)
        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(q_emb, k=k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({'chunk': self.chunks[idx], 'score': float(dist)})

        return results

    # ---------------------- Utilities ----------------------
    def set_active_document(self, doc_id: Optional[str]):
        self.active_doc_id = doc_id

    def list_documents(self) -> List[Dict]:
        return self.documents

    def get_active_document(self) -> Optional[Dict]:
        if not self.active_doc_id:
            return None
        for d in self.documents:
            if d['doc_id'] == self.active_doc_id:
                return d
        return None

    def save_index(self):
        if self.index is None or not self.index_file:
            return
        faiss.write_index(self.index, self.index_file)

    def load_index(self):
        if not self.index_file or not os.path.exists(self.index_file):
            return
        self.index = faiss.read_index(self.index_file)

