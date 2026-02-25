"""
Layer 3: Semantic Memory with Vector Embeddings

Stores facts and insights with semantic embeddings for similarity search.
Uses FAISS for efficient vector indexing.
"""

import json
import os
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Optional

from core.config import (
    MEMORY_FILE,
    SEMANTIC_CONFIG,
    BASE_DIR
)
from core.logger import logger


class SemanticMemory:
    """
    Vector-indexed semantic memory for facts and insights.
    Enables similarity-based retrieval.
    """

    def __init__(self, model_manager):
        """
        Initialize semantic memory.

        Args:
            model_manager: ModelManager instance for embeddings
        """
        self.model_manager = model_manager
        self.embeddings_file = os.path.join(BASE_DIR, "semantic_memory.json")
        self.index_file = os.path.join(BASE_DIR, "semantic_index.faiss")

        self.memories = []  # List of memory entries with metadata
        self.index = None   # FAISS index
        self.embedding_dim = None  # will be determined from stored or model embeddings

        self._load()

        # If still unknown, try to probe the model for embedding dimension
        if self.embedding_dim is None:
            try:
                sample = self.model_manager.embed([" "])[0]
                self.embedding_dim = len(sample)
            except Exception:
                # fallback to conservative default
                self.embedding_dim = 384
    def add_memory(self, text: str, metadata: Dict = None) -> None:
        """
        Add a memory entry with embedding.

        Args:
            text: Memory text
            metadata: Optional metadata dict
        """
        # Generate embedding

        embedding = self.model_manager.embed([text])[0]
        # ensure numpy float32 and consistent dimensionality
        embedding = np.array(embedding, dtype=np.float32)

        # If embedding dim differs from known, re-embed existing memories
        if self.embedding_dim is None:
            self.embedding_dim = embedding.shape[0]
        elif embedding.shape[0] != self.embedding_dim:
            logger.info(f"Embedding dim changed ({self.embedding_dim} -> {embedding.shape[0]}), re-embedding memories")
            self._reembed_all(embedding_dim=embedding.shape[0])
            self.embedding_dim = embedding.shape[0]

        # Create memory entry
        entry = {
            'id': len(self.memories),
            'text': text,
            'embedding': embedding.tolist(),
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }

        self.memories.append(entry)

        # Update or rebuild index
        if self.index is None:
            self._build_index()
        else:
            try:
                self.index.add(np.array([embedding], dtype=np.float32))
            except AssertionError as e:
                logger.warning(f"FAISS assertion during add: {e}, rebuilding index")
                self._build_index()
                self.index.add(np.array([embedding], dtype=np.float32))

        logger.debug(f"Added semantic memory (total: {len(self.memories)})")

    def search(
        self,
        query: str,
        k: int = 3,
        threshold: float = None
    ) -> List[str]:
        """
        Search semantic memory by similarity.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of most similar memory texts
        """
        if self.index is None or not self.memories:
            return []

        threshold = threshold or SEMANTIC_CONFIG.get('similarity_threshold', 0.5)

        # Get query embedding
        query_embedding = self.model_manager.embed([query])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # If query embedding dimension differs, try to re-embed stored memories
        if query_embedding.shape[0] != self.embedding_dim:
            logger.info(f"Query embedding dim {query_embedding.shape[0]} != stored dim {self.embedding_dim}, re-embedding memories")
            try:
                self._reembed_all(embedding_dim=query_embedding.shape[0])
                self.embedding_dim = query_embedding.shape[0]
            except Exception:
                logger.exception("Failed to re-embed memories for updated embedding dim")
                return []

        # Search
        k = min(k, len(self.memories))
        try:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k=k
            )
        except AssertionError as e:
            logger.warning(f"FAISS assertion during search: {e}, rebuilding index and retrying")
            self._build_index()
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k=k
            )

        # Filter by threshold and extract results
        results = []
        # `indices` is a 2D array; iterate over the first row and convert to ints
        for dist, idx_val in zip(distances[0], indices[0]):
            # FAISS uses L2 distance; convert to similarity
            similarity = 1.0 / (1.0 + dist)

            if similarity >= threshold:
                try:
                    idx = int(idx_val)
                    results.append(self.memories[idx]['text'])
                except Exception:
                    # skip invalid indices
                    continue

        logger.debug(f"Semantic search found {len(results)} results")
        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        Search with similarity scores.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (text, similarity_score) tuples
        """
        if self.index is None or not self.memories:
            return []

        query_embedding = self.model_manager.embed([query])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.shape[0] != self.embedding_dim:
            logger.info(f"Query embedding dim {query_embedding.shape[0]} != stored dim {self.embedding_dim}, re-embedding memories")
            try:
                self._reembed_all(embedding_dim=query_embedding.shape[0])
                self.embedding_dim = query_embedding.shape[0]
            except Exception:
                logger.exception("Failed to re-embed memories for updated embedding dim")
                return []

        k = min(k, len(self.memories))
        try:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k=k
            )
        except AssertionError as e:
            logger.warning(f"FAISS assertion during search: {e}, rebuilding index and retrying")
            self._build_index()
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k=k
            )

        results = []
        for dist, idx_val in zip(distances[0], indices[0]):
            similarity = 1.0 / (1.0 + dist)
            try:
                idx = int(idx_val)
                results.append((self.memories[idx]['text'], similarity))
            except Exception:
                continue

        return results

    def get_memory(self, memory_id: int) -> Optional[Dict]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory entry dict or None
        """
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id]
        return None

    def update_memory(self, memory_id: int, text: str) -> None:
        """
        Update a memory entry.

        Args:
            memory_id: ID of memory to update
            text: New text content
        """
        if 0 <= memory_id < len(self.memories):
            # Rebuild embedding
            embedding = self.model_manager.embed([text])[0]

            self.memories[memory_id]['text'] = text
            self.memories[memory_id]['embedding'] = embedding.tolist()
            self.memories[memory_id]['updated_at'] = datetime.now().isoformat()

            # Rebuild index
            self._build_index()
            logger.debug(f"Updated semantic memory {memory_id}")

    def delete_memory(self, memory_id: int) -> None:
        """
        Delete a memory entry.

        Args:
            memory_id: ID of memory to delete
        """
        if 0 <= memory_id < len(self.memories):
            del self.memories[memory_id]
            self._rebuild_ids()
            self._build_index()
            logger.debug(f"Deleted semantic memory {memory_id}")

    def _build_index(self) -> None:
        """Build FAISS index from all embeddings."""
        if not self.memories:
            self.index = None
            return

        embeddings = np.array(
            [m['embedding'] for m in self.memories],
            dtype=np.float32
        )

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        logger.debug(f"Built semantic index with {len(self.memories)} vectors")

    def _rebuild_ids(self) -> None:
        """Rebuild memory IDs after deletion."""
        for i, memory in enumerate(self.memories):
            memory['id'] = i

    # ================================================
    # PERSISTENCE
    # ================================================

    def save(self) -> None:
        """Save semantic memory to disk."""
        try:
            # Save memory entries
            with open(self.embeddings_file, 'w') as f:
                json.dump(self.memories, f, indent=2)

            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)

            logger.info(f"Saved {len(self.memories)} semantic memories")
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")

    def _load(self) -> None:
        """Load semantic memory from disk."""
        # Load embeddings
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r') as f:
                    self.memories = json.load(f)
                logger.info(f"Loaded {len(self.memories)} semantic memories")
                # determine embedding dim from stored memories if available
                if self.memories and 'embedding' in self.memories[0]:
                    try:
                        self.embedding_dim = len(self.memories[0]['embedding'])
                    except Exception:
                        self.embedding_dim = None
            except Exception as e:
                logger.warning(f"Failed to load semantic memories: {e}")

        # Load FAISS index
        if os.path.exists(self.index_file) and self.memories:
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info("Loaded FAISS semantic index")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, rebuilding: {e}")
                self._build_index()
        else:
            # Build index from loaded memories
            if self.memories:
                self._build_index()

    def _reembed_all(self, embedding_dim: int = None) -> None:
        """Recompute embeddings for all stored memories using current embedder.

        This is used when the embedding model or dimensionality changes.
        """
        if not self.memories:
            return

        texts = [m.get('text', '') for m in self.memories]
        try:
            new_embeddings = self.model_manager.embed(texts)
        except Exception as e:
            logger.error(f"Failed to re-embed memories: {e}")
            raise

        # validate and store
        for i, emb in enumerate(new_embeddings):
            arr = np.array(emb, dtype=np.float32)
            self.memories[i]['embedding'] = arr.tolist()

        # update embedding_dim if provided or inferred
        if embedding_dim is None:
            try:
                self.embedding_dim = len(new_embeddings[0])
            except Exception:
                self.embedding_dim = None
        else:
            self.embedding_dim = embedding_dim

        # rebuild index to reflect new embeddings
        self._build_index()

    # ================================================
    # STATISTICS AND INFO
    # ================================================

    def get_info(self) -> Dict:
        """Get semantic memory statistics."""
        return {
            'total_memories': len(self.memories),
            'embedding_dimension': self.embedding_dim,
            'index_exists': self.index is not None,
            'file_path': self.embeddings_file
        }

    def clear(self) -> None:
        """Clear all semantic memories."""
        self.memories = []
        self.index = None
        logger.info("Cleared all semantic memories")
