from datetime import datetime
from core.config import MAX_HISTORY_TURNS, STUDY_KEYWORDS
from memory.memory_store import MemoryStore
from retrieval.document_loader import load_documents
from retrieval.document_index import DocumentIndex
from core.model_manager import ModelManager


class AIService:

    def __init__(self):
        self.model_manager = ModelManager.get_instance()
        self.memory_store = MemoryStore()
        self.documents = load_documents()
        self.doc_index = DocumentIndex(self.documents)
        self.chat_history = []

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _is_personal_message(self, text):
        triggers = ["hello", "hi", "hey", "good morning", "good evening"]
        return any(t in text.lower() for t in triggers)

    def _is_study_query(self, text):
        return any(k in text.lower() for k in STUDY_KEYWORDS)

    def _extract_name(self):
        for item in self.memory_store.memory_items:
            text = item["text"].lower()
            if "my name is" in text:
                return item["text"].split("my name is")[-1].strip()
        return None

    def _extract_birth_year(self):
        for item in self.memory_store.memory_items:
            words = item["text"].split()
            for word in words:
                if word.isdigit() and len(word) == 4:
                    return int(word)
        return None

    def _calculate_age(self, birth_year, target_year=None):
        if not birth_year:
            return None

        current_year = datetime.now().year

        if target_year:
            return target_year - birth_year

        return current_year - birth_year

    # --------------------------------------------------
    # MAIN
    # --------------------------------------------------

    def ask(self, user_input):

        user_lower = user_input.lower()

        if self._is_personal_message(user_input):
            self.chat_history.clear()

        # Store memory triggers
        memory_triggers = [
            "my name is",
            "i was born",
            "my birthday is",
            "i like",
            "i struggle"
        ]

        if any(trigger in user_lower for trigger in memory_triggers):
            self.memory_store.add_memory(user_input)

        name = self._extract_name()
        birth_year = self._extract_birth_year()
        current_year = datetime.now().year
        age = self._calculate_age(birth_year)

        # Detect future year queries (e.g., "in 2030")
        target_year = None
        for word in user_lower.split():
            if word.isdigit() and len(word) == 4:
                target_year = int(word)

        future_age = None
        if birth_year and target_year:
            future_age = self._calculate_age(birth_year, target_year)

        # Semantic memory retrieval
        memory = self.memory_store.search(user_input)

        docs = ""
        if self._is_study_query(user_input):
            docs = self.doc_index.search(user_input)

        self.chat_history.append(f"User: {user_input}")
        history_text = "\n".join(
            self.chat_history[-MAX_HISTORY_TURNS * 2:]
        )

        # Prompt (simple & calm)
        prompt = f"""<s>[INST]
You are a helpful personal AI assistant.

System Facts:
- Current Year: {current_year}
- Name: {name}
- Birth Year: {birth_year}
- Current Age: {age}
- Future Age (if applicable): {future_age}

Use system facts when answering.
Be clear and natural.

Conversation:
{history_text}

User: {user_input}
[/INST]"""

        response = self.model_manager.generate(prompt)

        reply = response.strip()

        self.chat_history.append(f"Assistant: {reply}")

        return reply
