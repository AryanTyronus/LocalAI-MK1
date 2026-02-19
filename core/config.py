import os

MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MEMORY_FILE = os.path.join(BASE_DIR, "structured_memory.json")
KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge")

MAX_HISTORY_TURNS = 5
MEMORY_TOP_K = 3
DOC_TOP_K = 2

STUDY_KEYWORDS = [
    "physics", "math", "jee", "derive",
    "equation", "numerical", "solve",
    "formula", "calculate", "explain"
]
