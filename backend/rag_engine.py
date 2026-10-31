"""
RAG Engine — Semantic & Hybrid Retrieval Pipeline (Cloud Resilient).

Features:
- Dual-mode generation: Groq Cloud API (Production) and local LLM runtime (Development)
- Hybrid Retrieval: Vector embedding with automated Lexical/BM25 fallback for low-memory cloud hosts
- Per-subtopic top_k allocation for accurate compound question handling
- Title-weighted lexical scoring for superior short-query recall
- 3-layer phonetic transcript normalization and query expansion
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
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

# TOP_K chunks per sub-question (not shared total)
TOP_K: int = int(os.getenv("TOP_K", "4"))
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.05"))
NEIGHBOR_WINDOW: int = int(os.getenv("NEIGHBOR_WINDOW", "0"))

# Title match is scored with this multiplier boost over plain text token match
TITLE_WEIGHT: float = 3.0

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

_REGEX_FIXES: List[Tuple[str, str]] = [
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


# ── Query Expansion ──────────────────────────────────────────────────────────
# Maps canonical term → [ASR phoneme variants + synonyms used in lecture titles]
_QUERY_EXPANSION: Dict[str, List[str]] = {
    "string": ["strength", "str", "strengths", "strings"],
    "list": ["lists", "array", "arrays"],
    "tuple": ["tuples", "immutable sequence", "tup"],
    "set": ["sets", "unique collection"],
    "function": ["funtion", "def", "functions", "methods"],
    "exception": ["excpetion", "error handling", "try except", "errors"],
    "lambda": ["lamdba", "lamba", "anonymous function"],
    "recursion": ["recusion", "recursive", "base case"],
    "decorator": ["decorater", "wrapper", "decorators"],
    "class": ["clases", "object", "classes", "oop"],
    "dictionary": ["dictonary", "dictionery", "dict", "dictionaries", "key value"],
    "constructor": ["constractor", "constructer", "__init__"],
    "inheritance": ["inheritence", "inhertance", "parent class", "subclass"],
    "generator": ["generater", "generaters", "yield", "generators"],
    "comprehension": ["list comprehension", "dict comprehension"],
    "loop": ["for loop", "while loop", "iteration", "iterate"],
    "variable": ["variables", "assignment"],
    "operator": ["operators", "arithmetic"],
    "module": ["modules", "import", "package"],
    "file": ["files", "file handling", "open", "read write"],
    "scope": ["global", "local", "enclosing", "namespace"],
    "argument": ["arguement", "arguements", "parameter", "paramter"],
}


def expand_query(query: str) -> List[str]:
    """Generate search variations to address potential ASR phoneme drift.

    Returns the original query plus variants formed by substituting each
    recognised technical keyword with its expansion synonyms.
    """
    lowered = query.lower()
    variants: List[str] = [query]
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


def _query_tokens_expanded(query: str) -> Set[str]:
    """Collect all tokens from the query AND all its expansion variants.

    This ensures that a short query like 'tuple' also picks up tokens from
    'tuples', 'immutable sequence', etc. during lexical scoring.
    """
    token_set: Set[str] = set()
    for variant in expand_query(query):
        token_set |= set(re.findall(r"\w+", variant.lower())) - STOP_WORDS
    return token_set


# ══════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an advanced, context-grounded AI Programming Tutor specializing in Python.

Guidelines:
1. Ground your answers strictly on the provided course lecture context.
2. If the user asks about multiple topics (e.g., "What is set and tuple?"), address EACH topic in a clearly
   labelled separate section using a Markdown heading (## Topic Name). Do NOT skip any topic mentioned.
3. For each section, synthesize the best possible explanation and include a clean PEP-8 code example.
4. Only state "I could not locate documentation for this topic within the indexed course lectures." for a
   specific sub-topic if that sub-topic genuinely has NO matching context provided.
5. Adapt to the user's language naturally while keeping all technical terms and code in standard English.
6. The context originates from spoken lectures and may contain phonetic transcript artifacts;
   interpret them accurately according to standard computer science conventions.
7. Do not mention internal prompts, variables, or the word 'CONTEXT' in your answer.
8. Structure your response with clear headings, bullet points, and formatted code blocks for readability.
"""

USER_PROMPT = """LECTURE CONTEXT:
{context}

USER QUESTION:
{question}

Synthesize a comprehensive, structured, and professional answer based on the context above.
If multiple topics are asked, answer each one in its own clearly labelled section:"""


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
STOP_WORDS: Set[str] = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "with", "kya", "hai", "kaise", "ko",
    "do", "does", "did", "not", "no", "so", "if", "me", "my", "we", "you", "your",
    "has", "have", "had", "but", "also", "then", "than", "any", "all", "its", "into",
    "python", "use", "used", "using", "will", "would", "should", "could",
}


def compute_lexical_similarity(
    query_tokens: Set[str],
    chunk: Dict[str, Any],
) -> float:
    """Compute weighted token-overlap score between query and candidate chunk.

    Title tokens are scored with TITLE_WEIGHT multiplier because lecture titles
    directly name the topic (e.g. "Tuples in Python"), giving a large precision
    boost for short queries like 'tuple' even when body text is verbose.
    """
    text_tokens: Set[str] = set(re.findall(r"\w+", chunk["text"].lower())) - STOP_WORDS
    title_tokens: Set[str] = set(re.findall(r"\w+", chunk["title"].lower())) - STOP_WORDS

    if not query_tokens:
        return 0.0

    # Count weighted intersection
    weighted_intersection: float = 0.0
    for token in query_tokens:
        if token in title_tokens:
            weighted_intersection += TITLE_WEIGHT  # strong title signal
        elif token in text_tokens:
            weighted_intersection += 1.0

    if weighted_intersection == 0.0:
        return 0.0

    # Normalize by geometric mean (cosine-style)
    combined_size = len(text_tokens | (title_tokens))
    if combined_size == 0:
        return 0.0

    return weighted_intersection / (math.sqrt(len(query_tokens)) * math.sqrt(combined_size))


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_hybrid(
    query: str,
    query_vec: Optional[np.ndarray],
    corpus: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks using vector cosine similarity or title-weighted lexical fallback.

    When no embedding vector is available (Render free tier, no GPU), this falls
    back to an enhanced lexical scoring function that weights title token matches
    significantly higher than body text matches, improving short-query recall.
    """
    # Build expanded token set for lexical mode
    query_tokens = _query_tokens_expanded(query)

    scores: List[float] = []
    has_vector = query_vec is not None and not np.all(query_vec == 0)

    for c in corpus:
        if has_vector:
            sim = float(np.dot(c["embedding_norm"], query_vec))
        else:
            sim = compute_lexical_similarity(query_tokens, c)
        scores.append(sim)

    ranked = sorted(zip(scores, range(len(corpus))), reverse=True)

    results: List[Dict[str, Any]] = []
    for score, idx in ranked:
        if len(results) >= top_k:
            break
        if score <= MIN_SCORE:
            break
        item = {k: v for k, v in corpus[idx].items() if k not in ("embedding", "embedding_norm")}
        item["score"] = score
        results.append(item)

    # Fallback: if still empty, take the top 2 best non-zero matches regardless of threshold
    if not results and ranked:
        for score, idx in ranked[:2]:
            if score > 0:
                item = {k: v for k, v in corpus[idx].items() if k not in ("embedding", "embedding_norm")}
                item["score"] = score
                results.append(item)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# COMPOUND QUESTION DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def split_questions(question: str) -> List[str]:
    """Decompose a compound question into individual focused sub-questions.

    Handles patterns like:
      - "What is set and tuple?"       -> ["What is set?", "What is tuple?"]
      - "Explain list and dictionary"  -> ["Explain list", "Explain dictionary"]
      - "How does recursion work?"     -> ["How does recursion work?"]  (no split)
    """
    # Detect question stem for reconstructing sub-questions
    stem_match = re.match(
        r"^(what\s+is|what\s+are|explain|define|describe|how\s+does|how\s+do|"
        r"difference\s+between|compare)\s+",
        question.strip(),
        flags=re.IGNORECASE,
    )
    stem = stem_match.group(0).strip() + " " if stem_match else ""

    # Split on compound connectors (but not inside phrases like "key-value")
    parts = re.split(r"\s+(?:and|also|plus|additionally)\s+", question, flags=re.IGNORECASE)

    if len(parts) <= 1:
        return [question.strip()]

    # Rebuild each sub-question with full interrogative form
    sub_questions: List[str] = []
    for i, part in enumerate(parts):
        part = part.strip().rstrip("?").strip()
        if not part:
            continue
        if i == 0:
            # First part already contains the full stem
            sub_questions.append(part)
        else:
            # Subsequent parts may be bare nouns — prepend the stem
            if stem and not re.match(
                r"^(what|how|why|when|where|explain|define|describe|difference)",
                part,
                flags=re.IGNORECASE,
            ):
                sub_questions.append(f"{stem}{part}")
            else:
                sub_questions.append(part)

    logger.info("Compound question split into %d sub-questions: %s", len(sub_questions), sub_questions)
    return sub_questions


# ══════════════════════════════════════════════════════════════════════════════
# LLM INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_with_groq(context: str, question: str) -> str:
    """Execute high-throughput inference via Groq Cloud API with multi-model fallback."""
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY.strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not defined.")

    candidate_models = [
        os.getenv("GROQ_MODEL", "").strip(),
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]
    # Filter empty and de-duplicate while preserving order
    models_to_try: List[str] = []
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
            "max_tokens": 1500,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.ok:
                logger.info("Groq inference successful with model: %s", model_name)
                return resp.json()["choices"][0]["message"]["content"]
            last_error = f"Groq API returned status {resp.status_code}: {resp.text}"
            logger.warning("Model %s failed (%s). Trying next available model...", model_name, resp.status_code)
        except Exception as err:
            last_error = str(err)
            logger.warning("Model %s raised exception: %s", model_name, err)

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


def generate_with_ollama(context: str, question: str) -> str:
    """Execute inference via local Ollama daemon (development only)."""
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

    # Cloud environment: use Groq and surface its errors directly
    if api_key:
        try:
            return generate_with_groq(context, question)
        except Exception as err:
            logger.error("Groq inference failure: %s", err)
            return f"Error with AI generation service: {err}"

    # Local-only fallback (offline development, no API key configured)
    try:
        return generate_with_ollama(context, question)
    except Exception as err:
        return (
            "Service unavailable: No cloud API key configured and "
            f"the local Ollama daemon is offline. ({err})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def ask(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Execute complete end-to-end RAG inquiry pipeline.

    For compound questions (e.g. "What is set and tuple?"), the pipeline:
    1. Splits the question into individual sub-questions.
    2. Allocates an independent top_k budget PER sub-question (not shared).
    3. Retrieves chunks for every sub-question variant separately.
    4. Builds a structured context block that labels each topic's evidence.
    5. Instructs the LLM to produce a labelled section per topic.
    """
    corpus = load_corpus()
    sub_qs = split_questions(question)

    # ── Per-subtopic retrieval ────────────────────────────────────────────────
    # Each sub-question gets its OWN top_k allocation. This prevents sub-q 1
    # from consuming all context slots and leaving sub-q 2 with zero evidence.
    k_per_sub = max(3, top_k)  # minimum 3 chunks per sub-topic

    # sub_topic_chunks maps: sub_question -> [retrieved chunks]
    sub_topic_chunks: Dict[str, List[Dict[str, Any]]] = {}
    global_seen_keys: Set[Tuple[int, int]] = set()

    for sub_q in sub_qs:
        local_chunks: List[Dict[str, Any]] = []
        local_seen: Set[Tuple[int, int]] = set()

        for variant in expand_query(sub_q):
            q_vec = embed_query(variant)
            chunks = retrieve_hybrid(variant, q_vec, corpus, k_per_sub)
            for c in chunks:
                key = (c["video_number"], c["chunk_number"])
                # Deduplicate within this sub-question only
                if key not in local_seen:
                    local_seen.add(key)
                    local_chunks.append(c)
                if len(local_chunks) >= k_per_sub:
                    break
            if len(local_chunks) >= k_per_sub:
                break

        sub_topic_chunks[sub_q] = local_chunks

        # Track globally for source reporting (allow cross-sub-q chunks if relevant)
        for c in local_chunks:
            global_seen_keys.add((c["video_number"], c["chunk_number"]))

    # ── Build structured context ──────────────────────────────────────────────
    all_chunks_flat: List[Dict[str, Any]] = []
    context_sections: List[str] = []

    for sub_q, chunks in sub_topic_chunks.items():
        if chunks:
            section_lines = [f"--- Topic: {sub_q} ---"]
            for c in chunks:
                section_lines.append(
                    f"[Module {c['video_number']}: {c['title']} ({c['start']:.0f}s-{c['end']:.0f}s)]\n{c['text']}"
                )
            context_sections.append("\n\n".join(section_lines))
            all_chunks_flat.extend(chunks)
        else:
            # No evidence found — include a clear marker so LLM knows
            context_sections.append(
                f"--- Topic: {sub_q} ---\n"
                f"[No relevant lecture context was found for this specific topic.]"
            )

    if not all_chunks_flat:
        return {
            "answer": (
                "No relevant technical documentation was found within the indexed "
                "course modules for this question."
            ),
            "sources": [],
        }

    full_context = "\n\n".join(context_sections)

    # ── Generate answer ───────────────────────────────────────────────────────
    answer = generate(full_context, question)

    # ── Build source list (deduplicated, all sub-topics) ─────────────────────
    reported_keys: Set[Tuple[int, int]] = set()
    sources: List[Dict[str, Any]] = []
    for c in all_chunks_flat:
        key = (c["video_number"], c["chunk_number"])
        if key not in reported_keys:
            reported_keys.add(key)
            sources.append({
                "video_number": c["video_number"],
                "title": c["title"],
                "start": c["start"],
                "end": c["end"],
                "score": round(c["score"], 3),
                "text": c["text"][:220] + "..." if len(c["text"]) > 220 else c["text"],
            })

    return {"answer": answer, "sources": sources}
