from memory.memory_store import MemoryStore


class MemoryService:
    def __init__(self):
        self.store = MemoryStore()

    def add_memory(self, text: str):
        self.store.add_memory(text)

    def search(self, query: str):
        return self.store.search(query)