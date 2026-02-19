import os
from pypdf import PdfReader
from core.config import KNOWLEDGE_FOLDER
from core.logger import logger

def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_documents():
    documents = []

    if not os.path.exists(KNOWLEDGE_FOLDER):
        return documents

    for file in os.listdir(KNOWLEDGE_FOLDER):
        if file.endswith(".pdf"):
            logger.info(f"Loading PDF: {file}")
            reader = PdfReader(os.path.join(KNOWLEDGE_FOLDER, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.extend(split_text(text))

    return documents
