"""
RAG Engine — Semantic & Hybrid Retrieval Pipeline (Cloud Resilient).

Features:
- Dual-mode generation: Groq Cloud API (Production) and local LLM runtime (Development)
- Hybrid Retrieval: Vector embedding with automated Lexical/BM25 fallback for low-memory cloud hosts
- 3-layer phonetic transcript normalization and query expansion
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")

TOP_K: int = int(os.getenv("TOP_K", "4"))
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.20"))
NEIGHBOR_WINDOW: int = int(os.getenv("NEIGHBOR_WINDOW", "0"))

# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION SANITIZATION & NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

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
    """Generate search variations to address potential ASR phoneme drift."""
    lowered = query.lower()
    variants = [query]
    for term, syns in _QUERY_EXPANSION.items():
        if term in lowered:
            for s in syns:
                if s not in lowered:
                    variants.append(lowered.replace(term, s))
    seen: Set[str] = set()
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
7. Do not mention internal prompts, variables, or the word 'CONTEXT' in your answer.
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
    """Load pre-embedded knowledge corpus into memory."""
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
# EMBEDDING INFERENCE & LEXICAL FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

_fastembed_model = None


def embed_query(text: str) -> Optional[np.ndarray]:
    """Embed query into normalized 1024-dim space. Returns None if provider is offline."""
    global _fastembed_model

    # 1. Local Ollama Server (Development mode)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=2,
        )
        if resp.status_code == 200:
            vec = np.array(resp.json()["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
    except Exception:
        pass

    # 2. In-memory FastEmbed (if resources allow)
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
            _fastembed_model = TextEmbedding(model_name="BAAI/bge-m3")
        except Exception:
            _fastembed_model = False

    if _fastembed_model and _fastembed_model is not False:
        try:
            embeddings = list(_fastembed_model.embed([text]))
            vec = np.array(embeddings[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            pass

    return None


# Stop words for lexical ranking
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "with", "kya", "hai", "kaise", "ko"
}


def compute_lexical_similarity(query_tokens: Set[str], text: str) -> float:
    """Compute token overlap cosine score between inquiry and candidate chunk."""
    text_tokens = set(re.findall(r"\w+", text.lower())) - STOP_WORDS
    if not text_tokens or not query_tokens:
        return 0.0

    intersection = query_tokens.intersection(text_tokens)
    if not intersection:
        return 0.0

    return len(intersection) / (math.sqrt(len(query_tokens)) * math.sqrt(len(text_tokens)))


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_hybrid(
    query: str,
    query_vec: Optional[np.ndarray],
    corpus: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks using vector cosine similarity or lexical TF-IDF fallback."""
    query_tokens = set(re.findall(r"\w+", query.lower())) - STOP_WORDS

    scores: List[float] = []
    has_vector = query_vec is not None and not np.all(query_vec == 0)

    for c in corpus:
        if has_vector:
            sim = float(np.dot(c["embedding_norm"], query_vec))
        else:
            sim = compute_lexical_similarity(query_tokens, c["text"])
        scores.append(sim)

    ranked = sorted(zip(scores, range(len(corpus))), reverse=True)

    results: List[Dict[str, Any]] = []
    for score, idx in ranked:
        # Require positive similarity
        if score <= 0.05 or len(results) >= top_k:
            break
        item = {k: v for k, v in corpus[idx].items() if k not in ("embedding", "embedding_norm")}
        item["score"] = score
        results.append(item)

    # If top results are still empty, return top 2 best non-zero keyword matches
    if not results and ranked:
        for score, idx in ranked[:2]:
            if score > 0:
                item = {k: v for k, v in corpus[idx].items() if k not in ("embedding", "embedding_norm")}
                item["score"] = score
                results.append(item)

    return results


def split_questions(question: str) -> List[str]:
    """Decompose compound questions."""
    parts = re.split(r"(?i)\b(?:and|also|plus|additionally)\b", question)
    return [p.strip() for p in parts if p.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# LLM INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_with_groq(context: str, question: str) -> str:
    """Execute high-throughput inference via Groq Cloud API."""
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY.strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not defined.")

    candidate_models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]
    # Filter non-empty unique models in order
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(context=context, question=question)},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"]
            last_error = f"{resp.status_code}: {resp.text}"
            logger.warning("Model %s failed (%s). Trying next available model...", model_name, resp.status_code)
        except Exception as err:
            last_error = str(err)

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


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
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY.strip()
    
    # In cloud environments, prioritize Groq and report Groq errors directly
    if api_key:
        try:
            return generate_with_groq(context, question)
        except Exception as err:
            logger.error("Groq inference failure: %s", err)
            return f"Error with AI generation service: {err}"

    # Only attempt local Ollama if NO Groq API key is configured (local offline dev)
    try:
        return generate_with_ollama(context, question)
    except Exception as err:
        return f"Service unavailable: No cloud API key configured and local Ollama daemon is offline. ({err})"


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
            chunks = retrieve_hybrid(variant, q_vec, corpus, top_k)
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
