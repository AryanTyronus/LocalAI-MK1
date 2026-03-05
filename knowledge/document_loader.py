"""
Knowledge document loading and chunking.

Supports PDFs, markdown, text files, and repository source files.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Iterable

from pypdf import PdfReader

from core.config import BASE_DIR, KNOWLEDGE_FOLDER
from core.logger import logger


class KnowledgeDocumentLoader:
    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 200):
        self._chunk_chars = max(300, int(chunk_chars))
        self._overlap_chars = max(0, min(int(overlap_chars), self._chunk_chars // 2))

        self._code_dirs = ("core", "services", "memory", "retrieval", "tools", "tests")
        self._text_exts = {".md", ".txt", ".rst", ".json", ".yaml", ".yml"}
        self._code_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".sh"}
        self._ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules"}

    def load_chunks(self) -> List[Dict]:
        chunks: List[Dict] = []
        next_idx = 0

        # 1) Knowledge folder: PDFs + notes.
        if os.path.isdir(KNOWLEDGE_FOLDER):
            for root, dirs, files in os.walk(KNOWLEDGE_FOLDER):
                dirs[:] = [d for d in dirs if d not in self._ignore_dirs]
                for filename in files:
                    path = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    rel = os.path.relpath(path, BASE_DIR)
                    if ext == ".pdf":
                        file_chunks = self._load_pdf(path, rel)
                    elif ext in self._text_exts or ext in self._code_exts:
                        file_chunks = self._load_text(path, rel, source_type="knowledge")
                    else:
                        continue
                    for item in file_chunks:
                        item["id"] = next_idx
                        chunks.append(item)
                        next_idx += 1

        # 2) Repository code folders.
        for code_dir in self._code_dirs:
            abs_dir = os.path.join(BASE_DIR, code_dir)
            if not os.path.isdir(abs_dir):
                continue
            for root, dirs, files in os.walk(abs_dir):
                dirs[:] = [d for d in dirs if d not in self._ignore_dirs]
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in self._code_exts and ext not in self._text_exts:
                        continue
                    path = os.path.join(root, filename)
                    rel = os.path.relpath(path, BASE_DIR)
                    file_chunks = self._load_text(path, rel, source_type="repo")
                    for item in file_chunks:
                        item["id"] = next_idx
                        chunks.append(item)
                        next_idx += 1

        logger.info(f"KnowledgeDocumentLoader loaded {len(chunks)} chunks")
        return chunks

    def _stable_doc_id(self, rel_path: str) -> str:
        digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:10]
        return f"{rel_path}::{digest}"

    def _chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        out: List[str] = []
        start = 0
        step = self._chunk_chars - self._overlap_chars
        if step <= 0:
            step = self._chunk_chars
        while start < len(text):
            part = text[start:start + self._chunk_chars].strip()
            if part:
                out.append(part)
            start += step
        return out

    def _load_pdf(self, path: str, rel_path: str) -> List[Dict]:
        chunks: List[Dict] = []
        doc_id = self._stable_doc_id(rel_path)
        doc_name = os.path.basename(path)
        try:
            reader = PdfReader(path)
            for page_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                page_chunks = self._chunk_text(text)
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    chunks.append(
                        {
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "path": rel_path,
                            "page_number": page_idx,
                            "chunk_index": chunk_idx,
                            "text": chunk_text,
                            "source_type": "pdf",
                        }
                    )
        except Exception as exc:
            logger.warning(f"Failed to load PDF '{rel_path}': {exc}")
        return chunks

    def _load_text(self, path: str, rel_path: str, source_type: str) -> List[Dict]:
        chunks: List[Dict] = []
        doc_id = self._stable_doc_id(rel_path)
        doc_name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            text_chunks = self._chunk_text(text)
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "path": rel_path,
                        "page_number": 1,
                        "chunk_index": chunk_idx,
                        "text": chunk_text,
                        "source_type": source_type,
                    }
                )
        except Exception as exc:
            logger.warning(f"Failed to load text '{rel_path}': {exc}")
        return chunks

