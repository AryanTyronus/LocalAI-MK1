# ==========================================================
# PERSONAL LOCAL AI SYSTEM (MAC M4 OPTIMIZED)
# FINAL FIXED VERSION
# Semantic Memory + Stable Context + Study AI
# ==========================================================

import os
import json
import numpy as np
import requests
from bs4 import BeautifulSoup

from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# ================= SETTINGS =================

MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"

MEMORY_FILE = "structured_memory.json"
KNOWLEDGE_FOLDER = "knowledge"

MAX_HISTORY_TURNS = 5
MEMORY_TOP_K = 3
DOC_TOP_K = 2

# ================= LOAD MODEL =================

print("Loading AI model...")
model, tokenizer = load(MODEL_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# 1️⃣ SEMANTIC MEMORY SYSTEM
# =====================================================

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []

    data = json.load(open(MEMORY_FILE))

    # auto-fix old format (list of strings)
    if len(data) > 0 and isinstance(data[0], str):
        print("Old memory format detected. Converting...")

        new_memory = []
        for item in data:
            emb = embedder.encode(item)
            new_memory.append({
                "text": item,
                "embedding": emb.tolist()
            })

        json.dump(new_memory, open(MEMORY_FILE, "w"), indent=2)
        return new_memory

    return data


memory_items = load_memory()

def save_memory():
    json.dump(memory_items, open(MEMORY_FILE, "w"), indent=2)


def rebuild_memory_index():
    if not memory_items:
        return None

    embeddings = np.array([m["embedding"] for m in memory_items])
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index


memory_index = rebuild_memory_index()


memory_triggers = [
    "i like", "i prefer", "i struggle",
    "i find difficult", "my goal",
    "i want to improve", "i am preparing",
    "i don't understand", "i am weak at",
    "my name is"
]


def evolve_memory(user_input):
    global memory_index

    text = user_input.lower()

    if any(t in text for t in memory_triggers):

        embedding = embedder.encode(user_input)

        memory_items.append({
            "text": user_input,
            "embedding": embedding.tolist()
        })

        save_memory()
        memory_index = rebuild_memory_index()


def search_memory(query):
    if memory_index is None:
        return ""

    q_embed = embedder.encode([query])
    D, I = memory_index.search(
        np.array(q_embed),
        k=min(MEMORY_TOP_K, len(memory_items))
    )

    return "\n".join([memory_items[i]["text"] for i in I[0]])

# =====================================================
# 2️⃣ TEXT CHUNKING
# =====================================================

def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# =====================================================
# 3️⃣ LOAD PDF KNOWLEDGE
# =====================================================

documents = []

def load_pdfs():
    if not os.path.exists(KNOWLEDGE_FOLDER):
        return

    for file in os.listdir(KNOWLEDGE_FOLDER):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(KNOWLEDGE_FOLDER, file))
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            chunks = split_text(text)

            for chunk in chunks:
                documents.append(chunk)

load_pdfs()

# =====================================================
# 4️⃣ DOCUMENT SEARCH (FAISS)
# =====================================================

if documents:
    doc_embeddings = embedder.encode(documents)
    doc_index = faiss.IndexFlatL2(len(doc_embeddings[0]))
    doc_index.add(np.array(doc_embeddings))
else:
    doc_index = None


def search_docs(query):
    if doc_index is None:
        return ""

    q_embed = embedder.encode([query])
    D, I = doc_index.search(np.array(q_embed), k=DOC_TOP_K)

    return "\n".join([documents[i] for i in I[0]])

# =====================================================
# 5️⃣ INTERNET SEARCH (OPTIONAL)
# =====================================================

def internet_search(query):
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for a in soup.select(".result__snippet")[:3]:
            results.append(a.get_text())

        return "\n".join(results)

    except:
        return ""

# =====================================================
# 6️⃣ CONTEXT CONTROL
# =====================================================

def is_personal_message(text):
    triggers = [
        "my name is",
        "i am",
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening"
    ]
    return any(t in text.lower() for t in triggers)

# =====================================================
# MAIN AI FUNCTION
# =====================================================

chat_history = []

study_keywords = [
    "physics", "math", "jee", "derive",
    "equation", "numerical", "solve",
    "formula", "calculate", "explain"
]


def ask_ai(user_input):

    global chat_history

    evolve_memory(user_input)

    # ✅ FIX 3 — reset context on personal messages
    if is_personal_message(user_input):
        chat_history.clear()

    relevant_memory = search_memory(user_input)

    relevant_notes = ""
    if any(k in user_input.lower() for k in study_keywords):
        relevant_notes = search_docs(user_input)

    chat_history.append(f"User: {user_input}")

    history_text = "\n".join(chat_history[-MAX_HISTORY_TURNS*2:])

    prompt = f"""
You are a personal AI assistant.

If the user asks academic or study questions, act as a JEE-level physics and math tutor.
Otherwise respond normally and conversationally.

IMPORTANT:
- Answer ONLY the latest user message.
- Never continue previous answers unless asked.

Relevant User Memory:
{relevant_memory}

Relevant Study Material:
{relevant_notes}

Recent Conversation:
{history_text}

User: {user_input}
Assistant:
"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=300
    )

    reply = response.split("Assistant:")[-1].strip()

    chat_history.append(f"Assistant: {reply}")

    return reply

# =====================================================
# FLASK ENTRY
# =====================================================

def chat(user_input):
    return ask_ai(user_input)
