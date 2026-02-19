from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from core.config import MODEL_NAME
from core.logger import logger


class ModelManager:
    _instance = None

    def __init__(self):
        logger.info("Loading MLX model...")
        self.model, self.tokenizer = load(MODEL_NAME)

        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def generate(self, prompt, max_tokens=300):
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return output.strip()

    def embed(self, texts):
        return self.embedder.encode(texts)
