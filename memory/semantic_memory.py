"""
Layer 3: Semantic Memory with Vector Embeddings

Stores facts and insights with semantic embeddings for similarity search.
Uses FAISS for efficient vector indexing.

Features:
- Memory scoring with recency, usage, similarity, importance weights
- Semantic deduplication (>0.95 similarity prevents duplicate insertion)
- Automatic metadata tracking (created_at, last_accessed, access_count)
"""

import json
import os
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from core.config import (
    MEMORY_FILE,
    SEMANTIC_CONFIG,
    BASE_DIR
)
from core.logger import logger


# ============================================
# Memory Scoring Constants
# ============================================

# Scoring weights for memory prioritization
MEMORY_SCORING_WEIGHTS = {
    'similarity': 0.5,
    'recency': 0.2,
    'usage': 0.2,
    'importance': 0.1
}

# Deduplication threshold
DEDUPLICATION_SIMILARITY_THRESHOLD = 0.95

# Default importance for new memories
DEFAULT_MEMORY_IMPORTANCE = 0.5


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
    def add_memory(self, text: str, metadata: Dict = None) -> bool:
        """
        Add a memory entry with embedding.
        
        Performs deduplication check before adding.
        
        Args:
            text: Memory text
            metadata: Optional metadata dict
            
        Returns:
            True if memory was added, False if duplicate
        """
        # Check for duplicate before adding
        duplicate_info = self._check_duplicate(text)
        if duplicate_info['is_duplicate']:
            # Increment usage score of existing memory instead
            logger.info(f"Memory duplicate detected (similarity={duplicate_info['similarity']:.3f}), incrementing usage")
            self._increment_memory_access(duplicate_info['memory_id'])
            return False
        
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

        # Create memory entry with scoring metadata
        now = datetime.now().isoformat()
        entry = {
            'id': len(self.memories),
            'text': text,
            'embedding': embedding.tolist(),
            'metadata': metadata or {},
            'created_at': now,
            'last_accessed': now,
            'access_count': 1,
            'importance': metadata.get('importance', DEFAULT_MEMORY_IMPORTANCE) if metadata else DEFAULT_MEMORY_IMPORTANCE
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
        return True

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

    # ================================================
    # DEDUPLICATION
    # ================================================

    def _check_duplicate(self, text: str) -> Dict:
        """
        Check if text is duplicate of existing memory.
        
        Uses cosine similarity with threshold 0.95.
        
        Args:
            text: Text to check
            
        Returns:
            Dict with 'is_duplicate', 'similarity', 'memory_id'
        """
        if not self.memories or self.index is None:
            return {'is_duplicate': False, 'similarity': 0.0, 'memory_id': -1}
        
        try:
            # Get embedding for new text
            new_embedding = self.model_manager.embed([text])[0]
            new_embedding = np.array([new_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            new_norm = new_embedding / np.linalg.norm(new_embedding, axis=1, keepdims=True)
            
            # Search for similar memories
            k = min(3, len(self.memories))
            distances, indices = self.index.search(new_embedding, k=k)
            
            for dist, idx_val in zip(distances[0], indices[0]):
                similarity = 1.0 / (1.0 + dist)
                if similarity >= DEDUPLICATION_SIMILARITY_THRESHOLD:
                    return {
                        'is_duplicate': True,
                        'similarity': similarity,
                        'memory_id': int(idx_val)
                    }
            
            return {'is_duplicate': False, 'similarity': 0.0, 'memory_id': -1}
            
        except Exception as e:
            logger.warning(f"Error checking duplicate: {e}")
            return {'is_duplicate': False, 'similarity': 0.0, 'memory_id': -1}

    def _increment_memory_access(self, memory_id: int) -> None:
        """
        Increment access count and update last_accessed for a memory.
        
        Args:
            memory_id: ID of memory to update
        """
        if 0 <= memory_id < len(self.memories):
            self.memories[memory_id]['access_count'] = self.memories[memory_id].get('access_count', 0) + 1
            self.memories[memory_id]['last_accessed'] = datetime.now().isoformat()

    # ================================================
    # MEMORY SCORING
    # ================================================

    def compute_memory_score(
        self,
        memory: Dict,
        query_similarity: float = 0.0,
        current_time: datetime = None
    ) -> float:
        """
        Compute weighted score for a memory.
        
        Formula:
        final_score = similarity * 0.5
                    + recency * 0.2
                    + usage * 0.2
                    + importance * 0.1
        
        Args:
            memory: Memory entry dict
            query_similarity: Similarity score from query (0-1)
            current_time: Current time for recency calculation
            
        Returns:
            Final weighted score (0-1)
        """
        current_time = current_time or datetime.now()
        
        # Similarity score (already 0-1)
        similarity = query_similarity
        
        # Recency score (0-1, newer = higher)
        try:
            created = datetime.fromisoformat(memory.get('created_at', ''))
            age_seconds = (current_time - created).total_seconds()
            # Score decays over 30 days
            recency = max(0, 1.0 - (age_seconds / (30 * 24 * 3600)))
        except Exception:
            recency = 0.5
        
        # Usage score (0-1, more access = higher)
        access_count = memory.get('access_count', 1)
        usage = min(1.0, access_count / 10.0)  # Cap at 10 accesses
        
        # Importance score (0-1)
        importance = memory.get('importance', DEFAULT_MEMORY_IMPORTANCE)
        
        # Weighted combination
        final_score = (
            similarity * MEMORY_SCORING_WEIGHTS['similarity'] +
            recency * MEMORY_SCORING_WEIGHTS['recency'] +
            usage * MEMORY_SCORING_WEIGHTS['usage'] +
            importance * MEMORY_SCORING_WEIGHTS['importance']
        )
        
        return final_score

    def search_with_scoring(
        self,
        query: str,
        k: int = 3,
        threshold: float = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search semantic memory with weighted scoring.
        
        Returns memories sorted by combined score.
        
        Args:
            query: Search query
            k: Number of results
            threshold: Minimum similarity to consider
            
        Returns:
            List of (text, final_score, memory_dict) tuples
        """
        if self.index is None or not self.memories:
            return []
        
        threshold = threshold or SEMANTIC_CONFIG.get('similarity_threshold', 0.5)
        
        # Get query embedding
        query_embedding = self.model_manager.embed([query])[0]
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Handle dimension mismatch
        if query_embedding.shape[0] != self.embedding_dim:
            try:
                self._reembed_all(embedding_dim=query_embedding.shape[0])
                self.embedding_dim = query_embedding.shape[0]
            except Exception:
                logger.exception("Failed to re-embed for scoring search")
                return []
        
        # Search index
        k = min(k, len(self.memories))
        try:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k=k
            )
        except AssertionError as e:
            logger.warning(f"FAISS assertion in scoring search: {e}")
            self._build_index()
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                k=k
            )
        
        results = []
        for dist, idx_val in zip(distances[0], indices[0]):
            try:
                idx = int(idx_val)
                memory = self.memories[idx]
                
                # Convert L2 distance to similarity
                similarity = 1.0 / (1.0 + dist)
                
                if similarity >= threshold:
                    # Update access stats
                    self._increment_memory_access(idx)
                    
                    # Compute final weighted score
                    final_score = self.compute_memory_score(
                        memory, 
                        query_similarity=similarity
                    )
                    
                    results.append((memory['text'], final_score, memory))
            except Exception:
                continue
        
        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def get_top_memories(
        self,
        query: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Get top memories by weighted score.
        
        Wrapper around search_with_scoring that returns just text.
        
        Args:
            query: Search query
            top_k: Number of memories to return
            
        Returns:
            List of memory texts
        """
        results = self.search_with_scoring(query, k=top_k)
        return [text for text, score, _ in results]

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
        """Save semantic memory to disk atomically."""
        logger.debug(f"Starting semantic memory save - {len(self.memories)} memories")
        try:
            # Save memory entries atomically using temp file + rename
            temp_embeddings_file = self.embeddings_file + '.tmp'
            with open(temp_embeddings_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename for JSON
            os.replace(temp_embeddings_file, self.embeddings_file)
            logger.debug(f"Semantic memory JSON saved atomically to {self.embeddings_file}")

            # Save FAISS index (write to temp first, then rename)
            if self.index is not None:
                temp_index_file = self.index_file + '.tmp'
                faiss.write_index(self.index, temp_index_file)
                # Atomic rename for FAISS index
                os.replace(temp_index_file, self.index_file)
                logger.debug(f"FAISS index saved atomically to {self.index_file}")

            logger.info(f"Saved {len(self.memories)} semantic memories and FAISS index")
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")
            # Clean up temp files if they exist
            for temp_file in [self.embeddings_file + '.tmp', self.index_file + '.tmp']:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

    def _load(self) -> None:
        """Load semantic memory from disk."""
        logger.debug("Starting semantic memory load from disk")
        
        # Load embeddings
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r') as f:
                    self.memories = json.load(f)
                logger.info(f"Loaded {len(self.memories)} semantic memories from JSON")
                logger.debug(f"Semantic memory file path: {self.embeddings_file}")
                # determine embedding dim from stored memories if available
                if self.memories and 'embedding' in self.memories[0]:
                    try:
                        self.embedding_dim = len(self.memories[0]['embedding'])
                        logger.debug(f"Embedding dimension from loaded memories: {self.embedding_dim}")
                    except Exception:
                        self.embedding_dim = None
            except Exception as e:
                logger.warning(f"Failed to load semantic memories: {e}")

        # Load FAISS index
        if os.path.exists(self.index_file) and self.memories:
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS semantic index with {self.index.ntotal} vectors")
                logger.debug(f"FAISS index file path: {self.index_file}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, rebuilding: {e}")
                self._build_index()
        else:
            # Build index from loaded memories
            if self.memories:
                self._build_index()
                logger.debug("Built FAISS index from loaded memories")

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
