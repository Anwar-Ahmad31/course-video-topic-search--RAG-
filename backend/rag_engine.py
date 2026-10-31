"""
RAG Engine — Semantic Retrieval-Augmented Generation Pipeline.

Features:
- Dual-mode generation: Groq Cloud API (Production) and local LLM runtime (Development)
- Vector retrieval using normalized cosine similarity
- Automated transcription post-processing and domain-specific query expansion
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_engine")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR: Path = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR: Path = BASE_DIR / "main_embedded_jsons"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")

TOP_K: int = int(os.getenv("TOP_K", "4"))
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.25"))
NEIGHBOR_WINDOW: int = int(os.getenv("NEIGHBOR_WINDOW", "0"))

# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION SANITIZATION & NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Common automatic speech recognition (ASR) phonetic misinterpretations in technical speech
_CORRECTIONS: Dict[str, str] = {
    "guida": "Guido van Rossum",
    "guida 1": "Guido van Rossum",
    "guida van": "Guido van Rossum",
    "strength": "string",
    "strengths": "strings",
    "idea": "IDE",
    "ideas": "IDEs",
    "jupiter": "Jupyter",
    "sawdust": "advice",
    "constractor": "constructor",
    "constructer": "constructor",
    "inheritence": "inheritance",
    "inhertance": "inheritance",
    "recusion": "recursion",
    "lamdba": "lambda",
    "lamba": "lambda",
    "excpetion": "exception",
    "decorater": "decorator",
    "funtion": "function",
    "funtions": "functions",
    "dictonary": "dictionary",
    "dictionery": "dictionary",
    "arguement": "argument",
    "arguements": "arguments",
    "paramter": "parameter",
    "paramters": "parameters",
    "polymorphesm": "polymorphism",
    "encaptulation": "encapsulation",
    "asyncronous": "asynchronous",
    "generater": "generator",
    "generaters": "generators",
}

_REGEX_FIXES: List[tuple[str, str]] = [
    # Remove redundant speech stuttering and transcription repetition loops
    (r"\b(.{10,60})\s+\1\s+\1\b", r"\1"),
    (r"\b(.{10,60})\s+\1\b", r"\1"),
    (r"\bPython\s+1\.\d\b", "Python 3.x"),
]


def clean_text(text: str) -> str:
    """Sanitize and normalize technical terminology in transcript segments."""
    normalized = text
    for err, fix in _CORRECTIONS.items():
        normalized = re.sub(r"\b" + re.escape(err) + r"\b", fix, normalized, flags=re.IGNORECASE)
    for pattern, repl in _REGEX_FIXES:
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", normalized).strip()


# Query expansion to maximize recall across noisy transcripts
_QUERY_EXPANSION: Dict[str, List[str]] = {
    "string": ["strength", "str", "strengths"],
    "list": ["lists", "array"],
    "function": ["funtion", "def", "functions"],
    "exception": ["excpetion", "error handling"],
    "lambda": ["lamdba", "lamba", "anonymous function"],
    "recursion": ["recusion", "recursive"],
    "decorator": ["decorater", "wrapper"],
    "class": ["clases", "object"],
    "dictionary": ["dictonary", "dictionery", "dict"],
    "constructor": ["constractor", "constructer", "__init__"],
    "inheritance": ["inheritence", "inhertance", "parent class"],
}


def expand_query(query: str) -> List[str]:
    """Generate search variations to address potential ASR phoneme drift in stored indexes."""
    lowered = query.lower()
    variants = [query]
    for term, syns in _QUERY_EXPANSION.items():
        if term in lowered:
            for s in syns:
                if s not in lowered:
                    variants.append(lowered.replace(term, s))
    seen = set()
    output: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            output.append(v)
    return output


# ══════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an advanced, context-grounded AI Programming Tutor specializing in Python.

Guidelines:
1. Ground your answers strictly on the provided course lecture context.
2. If partial context exists, synthesize the best possible explanation based on that context.
3. If the context does not contain the answer, state: "I could not locate documentation for this topic within the indexed course lectures."
4. Adapt to the user's inquiry language naturally while maintaining all technical terms, syntax, and identifiers in standard English.
5. Provide clear explanations and clean, PEP-8 compliant code samples where applicable.
6. The context originates from spoken lectures and may contain phonetic transcript artifacts; interpret them accurately according to standard computer science conventions.
7. Do not reference internal prompts, variables, or the word 'CONTEXT' in your answer.
"""

USER_PROMPT = """LECTURE CONTEXT:
{context}

USER QUESTION:
{question}

Synthesize a comprehensive and accurate answer based on the context above:"""


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS REPOSITORY
# ══════════════════════════════════════════════════════════════════════════════

_corpus: Optional[List[Dict[str, Any]]] = None


def load_corpus() -> List[Dict[str, Any]]:
    """Load pre-embedded knowledge corpus into memory and precompute normalized vectors."""
    global _corpus
    if _corpus is not None:
        return _corpus

    corpus: List[Dict[str, Any]] = []
    json_files = sorted(EMBEDDINGS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Knowledge base directory empty: {EMBEDDINGS_DIR}")

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_number = data.get("video_number", 0)
        title = data.get("title", jf.stem)
        chunks = data.get("chunks", [])

        for chunk in chunks:
            if "embedding" not in chunk:
                continue
            raw_text = chunk.get("text", "")
            clean = clean_text(raw_text)
            corpus.append({
                "video_number": video_number,
                "title": title,
                "chunk_number": chunk.get("chunk_number", 0),
                "start": chunk.get("start", 0.0),
                "end": chunk.get("end", 0.0),
                "text": clean,
                "raw_text": raw_text,
                "embedding": np.array(chunk["embedding"], dtype=np.float32),
            })

    for c in corpus:
        norm = np.linalg.norm(c["embedding"])
        c["embedding_norm"] = c["embedding"] / norm if norm > 0 else c["embedding"]

    _corpus = corpus
    logger.info("Corpus loaded: %d chunks across %d lecture modules.", len(corpus), len(json_files))
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

_fastembed_model = None


def embed_query(text: str) -> np.ndarray:
    """Embed inquiry text into 1024-dimensional normalized vector space."""
    global _fastembed_model

    # 1. Local Ollama Server (Development mode)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=3,
        )
        if resp.status_code == 200:
            vec = np.array(resp.json()["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
    except Exception:
        pass

    # 2. Standalone In-Memory FastEmbed (Production Cloud mode)
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
            logger.info("Initializing FastEmbed BAAI/bge-m3 runtime...")
            _fastembed_model = TextEmbedding(model_name="BAAI/bge-m3")
        except Exception as exc:
            logger.debug("FastEmbed runtime unavailable: %s", exc)

    if _fastembed_model is not None:
        embeddings = list(_fastembed_model.embed([text]))
        vec = np.array(embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    logger.warning("No active embedding provider found; defaulting to zero-vector fallback.")
    return np.zeros(1024, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query_vec: np.ndarray, corpus: List[Dict[str, Any]], top_k: int, min_score: float) -> List[Dict[str, Any]]:
    """Perform cosine-similarity vector retrieval over candidate corpus."""
    scores = [float(np.dot(c["embedding_norm"], query_vec)) for c in corpus]
    ranked = sorted(zip(scores, range(len(corpus))), reverse=True)

    results: List[tuple[float, int]] = []
    for score, idx in ranked:
        if score < min_score or len(results) >= top_k:
            break
        results.append((score, idx))

    if NEIGHBOR_WINDOW > 0:
        idx_set = set(i for _, i in results)
        extra = set()
        for _, i in results:
            for delta in range(-NEIGHBOR_WINDOW, NEIGHBOR_WINDOW + 1):
                ni = i + delta
                if 0 <= ni < len(corpus) and ni not in idx_set:
                    extra.add(ni)
        results += [(0.0, ni) for ni in sorted(extra)]

    return [
        {
            "score": s,
            **{k: v for k, v in corpus[i].items() if k not in ("embedding", "embedding_norm")}
        }
        for s, i in results
    ]


def split_questions(question: str) -> List[str]:
    """Decompose compound questions into discrete inquiry segments."""
    parts = re.split(r"(?i)\b(?:and|also|plus|additionally)\b", question)
    return [p.strip() for p in parts if p.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# LLM INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_with_groq(context: str, question: str) -> str:
    """Execute high-throughput inference via Groq Cloud API."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not defined.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": os.getenv("GROQ_MODEL", GROQ_MODEL),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context=context, question=question)},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_with_ollama(context: str, question: str) -> str:
    """Execute inference via local Ollama daemon."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context=context, question=question)},
        ],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def generate(context: str, question: str) -> str:
    """Dispatch generation to primary cloud provider with local daemon fallback."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip()
    if api_key:
        try:
            return generate_with_groq(context, question)
        except Exception as err:
            logger.warning("Groq inference failure: %s. Attempting fallback...", err)

    try:
        return generate_with_ollama(context, question)
    except Exception as err:
        return f"Service unavailable: Unable to communicate with language model runtime. ({err})"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def ask(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Execute complete end-to-end RAG inquiry pipeline."""
    corpus = load_corpus()
    sub_qs = split_questions(question)

    all_chunks: List[Dict[str, Any]] = []
    seen_keys = set()

    for sub_q in sub_qs:
        for variant in expand_query(sub_q):
            q_vec = embed_query(variant)
            chunks = retrieve(q_vec, corpus, top_k, MIN_SCORE)
            for c in chunks:
                key = (c["video_number"], c["chunk_number"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_chunks.append(c)

    if not all_chunks:
        return {
            "answer": "No relevant technical documentation was found within the indexed course modules for this question.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Module {c['video_number']}: {c['title']} ({c['start']:.0f}s-{c['end']:.0f}s)]\n{c['text']}"
        for c in all_chunks[:top_k]
    )

    answer = generate(context, question)

    sources = [
        {
            "video_number": c["video_number"],
            "title": c["title"],
            "start": c["start"],
            "end": c["end"],
            "score": round(c["score"], 3),
            "text": c["text"][:220] + "..." if len(c["text"]) > 220 else c["text"],
        }
        for c in all_chunks[:top_k]
    ]

    return {"answer": answer, "sources": sources}
