import numpy as np
import faiss

from core.model_manager import ModelManager
from core.config import DOC_TOP_K

class DocumentIndex:

    def __init__(self, documents):
        self.model_manager = ModelManager.get_instance()
        self.documents = documents
        self.index = None
        self._build_index()

    def _build_index(self):
        if not self.documents:
            return

        embeddings = self.model_manager.embed(self.documents)
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))

    def search(self, query):
        if self.index is None:
            return ""

        query_vec = self.model_manager.embed([query])
        D, I = self.index.search(
            np.array(query_vec),
            k=DOC_TOP_K
        )

        return "\n".join(
            [self.documents[i] for i in I[0]]
        )
