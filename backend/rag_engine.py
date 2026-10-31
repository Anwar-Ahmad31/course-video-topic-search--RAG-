"""
RAG Engine — Production Semantic & BM25 Hybrid Retrieval Pipeline.

Features:
- Dual-mode generation: Groq Cloud API (Production) and local LLM runtime (Development)
- BM25 Okapi with Title Weighting: State-of-the-art lexical ranking optimized for technical documentation
- Phrase boost: Rewards exact multi-word technical phrase matches (e.g., 'for loop', 'virtual environment')
- Noise Fragment Filter: Prunes Whisper ASR hallucination loops and sub-8-word silence artifacts
- Intelligent Title Cleaning: Strips course-wide boilerplate suffixes to reveal true module subjects
- Per-subtopic retrieval allocation: Solves compound question starvation
- Zero hardcoding: Completely data-driven retrieval over the full 52-module lecture corpus
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
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

TOP_K: int = int(os.getenv("TOP_K", "4"))

# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT SANITIZATION & TITLE NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

_PHONETIC_REPLACEMENTS: Dict[str, str] = {
    "guida": "Guido van Rossum",
    "guida 1": "Guido van Rossum",
    "guida van": "Guido van Rossum",
    "jupiter": "Jupyter",
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
}


def clean_text(text: str) -> str:
    """Sanitize technical terminology and eliminate Whisper ASR repetition loops."""
    normalized = text

    # Remove consecutive repeated words caused by ASR silence looping (e.g., 'Python. Python. Python.')
    normalized = re.sub(r"(\b\w+[\.\,]?\s*)\1{2,}", r"\1", normalized, flags=re.IGNORECASE)

    # Phonetic corrections
    for err, fix in _PHONETIC_REPLACEMENTS.items():
        normalized = re.sub(r"\b" + re.escape(err) + r"\b", fix, normalized, flags=re.IGNORECASE)

    return re.sub(r"\s{2,}", " ", normalized).strip()


def clean_title(title: str) -> str:
    """Normalize lecture module title by stripping course-wide boilerplate suffixes."""
    # 1. Strip the YouTube series suffix '| Python Tutorial' or '? Python Tutorial'
    cleaned = re.sub(r"\s*[\uff5c|?]\s*Python\s+Tutorial.*$", "", title, flags=re.IGNORECASE).strip()
    # 2. Strip 'in Python' suffix if it's just a language qualifier (e.g., 'Strings in Python' -> 'Strings')
    # but preserve titles where Python is the primary topic (e.g., 'Introduction to Programming & Python')
    if not re.search(r"(&\s*Python|About\s+Python|Programming\s*&\s*Python)", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"\s+in\s+Python\b", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


# Conversational question words to down-weight during keyword extraction
CONVERSATIONAL_WORDS: Set[str] = {
    "what", "is", "are", "was", "were", "how", "does", "do", "did", "why",
    "when", "where", "which", "who", "whom", "whose", "can", "could", "would",
    "should", "will", "shall", "explain", "describe", "define", "tell", "me",
    "about", "give", "show", "example", "examples", "difference", "between",
    "kya", "hai", "kaise", "batao", "mujhe", "samjhao", "the", "a", "an",
    "in", "of", "to", "for", "with", "on", "at", "by", "from",
}


def extract_search_terms(query: str) -> List[str]:
    """Extract informative keywords from query, falling back to full tokens if needed."""
    raw_tokens = tokenize(query)
    filtered = [t for t in raw_tokens if t not in CONVERSATIONAL_WORDS]
    return filtered if filtered else raw_tokens


# ══════════════════════════════════════════════════════════════════════════════
# BM25 OKAPI ENGINE (Zero hardcoding, completely data-driven)
# ══════════════════════════════════════════════════════════════════════════════

class BM25Index:
    """In-memory BM25 Okapi index over all lecture transcript chunks."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 4.0,
    ) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        self.N = len(corpus)

        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.df: Counter[str] = Counter()

        for c in corpus:
            body_tokens = tokenize(c["text"])
            title_tokens = tokenize(c.get("clean_title", c["title"]))

            weighted_tokens = body_tokens + (title_tokens * int(title_weight))
            self.doc_lens.append(len(weighted_tokens))

            tf = Counter(weighted_tokens)
            self.doc_term_freqs.append(tf)

            for term in set(weighted_tokens):
                self.df[term] += 1

        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

        # Precompute Robertson-Spärck Jones IDF
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))

    def score_query(self, query: str) -> List[float]:
        """Compute BM25 relevance scores for all chunks."""
        terms = extract_search_terms(query)
        if not terms:
            terms = tokenize(query)

        scores = [0.0] * self.N
        if not terms:
            return scores

        for term in terms:
            idf_val = self.idf.get(term, 0.0)
            if idf_val <= 0.0:
                continue

            for idx, tf_dict in enumerate(self.doc_term_freqs):
                if term in tf_dict:
                    freq = tf_dict[term]
                    doc_len = self.doc_lens[idx]
                    num = freq * (self.k1 + 1.0)
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[idx] += idf_val * (num / denom)

        # Phrase bonus: reward chunks where adjacent query terms appear together
        if len(terms) >= 2:
            bigrams = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms) - 1)]
            for idx, c in enumerate(self.corpus):
                c_text_low = c["text"].lower()
                c_title_low = c["title"].lower()
                for bg in bigrams:
                    if bg in c_title_low:
                        scores[idx] += 8.0
                    elif bg in c_text_low:
                        scores[idx] += 3.0

        return scores


# ══════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert, context-grounded AI Programming Tutor specializing in Python.

Guidelines:
1. Ground your answers strictly on the provided course lecture context.
2. If the user asks about multiple topics (e.g., "What is set and tuple?"), address EACH topic in a clearly
   labelled separate section using a Markdown heading (## Topic Name). Do NOT omit any topic.
3. For each section, synthesize a clear explanation and include clean PEP-8 compliant code examples.
4. Only state "I could not locate documentation for this topic within the indexed course lectures." for a
   specific sub-topic if that sub-topic genuinely has NO matching context provided.
5. Adapt to the user's language naturally while keeping all technical terms and code in standard English.
6. The context originates from spoken lectures and may contain phonetic transcript artifacts;
   interpret them accurately according to standard computer science conventions.
7. Do not mention internal prompts, variables, or the word 'CONTEXT' in your answer.
8. Structure your response with clear headings, bullet points, and formatted code blocks for maximum readability.
"""

USER_PROMPT = """LECTURE CONTEXT:
{context}

USER QUESTION:
{question}

Synthesize a comprehensive, structured, and professional answer based on the context above:"""


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS REPOSITORY & INDEXING
# ══════════════════════════════════════════════════════════════════════════════

_corpus: Optional[List[Dict[str, Any]]] = None
_bm25_index: Optional[BM25Index] = None


def load_corpus() -> List[Dict[str, Any]]:
    """Load pre-embedded knowledge corpus into memory and initialize BM25 index."""
    global _corpus, _bm25_index
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
        raw_title = data.get("title", jf.stem)
        clean_t = clean_title(raw_title)
        chunks = data.get("chunks", [])

        for chunk in chunks:
            if "embedding" not in chunk:
                continue
            raw_text = chunk.get("text", "")
            cleaned_text = clean_text(raw_text)

            # Filter out non-speech silence noise fragments (chunks with fewer than 8 words or 5 unique words)
            words = tokenize(cleaned_text)
            if len(words) < 8 or len(set(words)) < 5:
                continue

            corpus.append({
                "video_number": video_number,
                "title": raw_title,
                "clean_title": clean_t,
                "chunk_number": chunk.get("chunk_number", 0),
                "start": chunk.get("start", 0.0),
                "end": chunk.get("end", 0.0),
                "text": cleaned_text,
                "raw_text": raw_text,
                "embedding": np.array(chunk["embedding"], dtype=np.float32),
            })

    for c in corpus:
        norm = np.linalg.norm(c["embedding"])
        c["embedding_norm"] = c["embedding"] / norm if norm > 0 else c["embedding"]

    _corpus = corpus
    _bm25_index = BM25Index(corpus)
    logger.info("Corpus loaded: %d clean chunks across %d lecture modules. BM25 index built.", len(corpus), len(json_files))
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING INFERENCE
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


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL (Vector + BM25 Fusion)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_hybrid(
    query: str,
    query_vec: Optional[np.ndarray],
    corpus: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks using BM25 ranking fused with vector cosine similarity (if available)."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index(corpus)

    bm25_scores = _bm25_index.score_query(query)
    max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
    norm_bm25 = [s / max_bm25 for s in bm25_scores]

    has_vector = query_vec is not None and not np.all(query_vec == 0)

    final_scores: List[float] = []
    for idx, c in enumerate(corpus):
        if has_vector:
            cos_sim = float(np.dot(c["embedding_norm"], query_vec))
            score = 0.60 * max(0.0, cos_sim) + 0.40 * norm_bm25[idx]
        else:
            score = norm_bm25[idx]
        final_scores.append(score)

    ranked = sorted(zip(final_scores, range(len(corpus))), reverse=True)

    results: List[Dict[str, Any]] = []
    for score, idx in ranked:
        if len(results) >= top_k:
            break
        if score <= 0.01:
            break
        item = {k: v for k, v in corpus[idx].items() if k not in ("embedding", "embedding_norm")}
        item["score"] = score
        results.append(item)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# COMPOUND QUESTION DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def split_questions(question: str) -> List[str]:
    """Decompose compound questions cleanly while preserving question intent for each topic."""
    stem_match = re.match(
        r"^(what\s+is|what\s+are|explain|define|describe|how\s+does|how\s+do|"
        r"difference\s+between|compare)\s+",
        question.strip(),
        flags=re.IGNORECASE,
    )
    stem = stem_match.group(0).strip() + " " if stem_match else ""

    parts = re.split(r"\s+(?:and|also|plus|additionally)\s+", question, flags=re.IGNORECASE)

    if len(parts) <= 1:
        return [question.strip()]

    sub_questions: List[str] = []
    for i, part in enumerate(parts):
        part = part.strip().rstrip("?").strip()
        if not part:
            continue
        if i == 0:
            sub_questions.append(part)
        else:
            if stem and not re.match(
                r"^(what|how|why|when|where|explain|define|describe|difference)",
                part,
                flags=re.IGNORECASE,
            ):
                sub_questions.append(f"{stem}{part}")
            else:
                sub_questions.append(part)

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

    if api_key:
        try:
            return generate_with_groq(context, question)
        except Exception as err:
            logger.error("Groq inference failure: %s", err)
            return f"Error with AI generation service: {err}"

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
    """Execute complete end-to-end RAG inquiry pipeline with BM25 + Hybrid retrieval."""
    corpus = load_corpus()
    sub_qs = split_questions(question)

    k_per_sub = max(3, top_k)
    sub_topic_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for sub_q in sub_qs:
        q_vec = embed_query(sub_q)
        chunks = retrieve_hybrid(sub_q, q_vec, corpus, k_per_sub)
        sub_topic_chunks[sub_q] = chunks

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
    answer = generate(full_context, question)

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
