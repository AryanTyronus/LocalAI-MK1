import json
import os
import numpy as np
import faiss

from core.config import MEMORY_FILE, MEMORY_TOP_K
from core.logger import logger
from core.model_manager import ModelManager

class MemoryStore:

    def __init__(self):
        self.model_manager = ModelManager.get_instance()
        self.memory_items = []
        self.index = None
        self._load_memory()

    def _load_memory(self):
        if not os.path.exists(MEMORY_FILE):
            self.memory_items = []
            return

        with open(MEMORY_FILE, "r") as f:
            self.memory_items = json.load(f)

        self._build_index()

    def _build_index(self):
        if not self.memory_items:
            self.index = None
            return

        embeddings = np.array([m["embedding"] for m in self.memory_items])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def save(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory_items, f, indent=2)

    def add_memory(self, text):
        embedding = self.model_manager.embed([text])[0]

        self.memory_items.append({
            "text": text,
            "embedding": embedding.tolist()
        })

        if self.index is None:
            self._build_index()
        else:
            self.index.add(np.array([embedding]))

        self.save()

    def search(self, query):
        if self.index is None:
            return ""

        query_vec = self.model_manager.embed([query])
        D, I = self.index.search(
            np.array(query_vec),
            k=min(MEMORY_TOP_K, len(self.memory_items))
        )

        return "\n".join(
            [self.memory_items[i]["text"] for i in I[0]]
        )
