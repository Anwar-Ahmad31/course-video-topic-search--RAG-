"""
text_cleaner.py
───────────────
3-layer fix for Whisper transcription errors:
  Layer 1 – Static correction dictionary (known errors → correct terms)
  Layer 2 – Regex pattern fixes (repeated phrases, garbled patterns)
  Layer 3 – LLM-based context-aware cleaning (optional, for retrieval)
"""

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# LAYER 1: Known wrong → correct word mapping
# Add more entries as you discover errors in your transcripts
# ─────────────────────────────────────────────────────────────────
CORRECTION_DICT: dict[str, str] = {
    # Python creator name
    "guida":               "Guido van Rossum",
    "guida van":           "Guido van Rossum",
    "guida 1":             "Guido van Rossum",
    "guide":               "Guido van Rossum",

    # IDEs / Tools
    "idea":                "IDE",
    "ideas":               "IDEs",
    "collaborative browser based idea": "Google Colab (browser-based IDE)",
    "collab":              "Google Colab",
    "jupiter":             "Jupyter",
    "jupyter notebook":    "Jupyter Notebook",

    # Python terms commonly garbled
    "strength":            "string",
    "strength methods":    "string methods",
    "strengths":           "strings",
    "sawdust":             "advice",
    "bite size":           "bitewise",
    "integer numbers":     "integers",
    "float numbers":       "float",
    "boolean":             "boolean",
    "tuple":               "tuple",
    "dictonary":           "dictionary",
    "dictionery":          "dictionary",
    "dctionary":           "dictionary",
    "funtion":             "function",
    "funtions":            "functions",
    "defenition":          "definition",
    "arguement":           "argument",
    "arguements":          "arguments",
    "paramter":            "parameter",
    "paramters":           "parameters",
    "inheritence":         "inheritance",
    "inhertance":          "inheritance",
    "encaptulation":       "encapsulation",
    "polymorphesm":        "polymorphism",
    "polymorphism":        "polymorphism",
    "recusion":            "recursion",
    "recusion":            "recursion",
    "excpetion":           "exception",
    "exection":            "exception",
    "excpetion handling":  "exception handling",
    "lamdba":              "lambda",
    "lamba":               "lambda",
    "decorater":           "decorator",
    "decoraters":          "decorators",
    "generater":           "generator",
    "generaters":          "generators",
    "asyncronous":         "asynchronous",
    "asyncio":             "asyncio",
    "pip install":         "pip install",
    "virtual envirnoment": "virtual environment",
    "virtual enviroment":  "virtual environment",
    "venv":                "virtual environment (venv)",

    # OOP terms
    "object oriented":     "object-oriented",
    "oops":                "OOP",
    "oop":                 "OOP",
    "clases":              "classes",
    "classs":              "class",
    "instence":            "instance",
    "instences":           "instances",
    "constractor":         "constructor",
    "constructer":         "constructor",

    # Hinglish common patterns
    "isko":                "this",
    "matlab":              "meaning",
    "mane":                "means",
    "matlab hai":          "means",
    "dekho":               "see",
    "samjho":              "understand",
    "ye hai":              "this is",
    "python mein":         "in Python",
    "code karo":           "write code",
}

# ─────────────────────────────────────────────────────────────────
# LAYER 2: Regex pattern fixes
# ─────────────────────────────────────────────────────────────────
REGEX_PATTERNS: list[tuple[str, str]] = [
    # Remove repeated phrases (e.g., "we will get we will get we will get")
    (r'\b(.{10,60})\s+\1\s+\1\b', r'\1'),
    (r'\b(.{10,60})\s+\1\b',       r'\1'),

    # Fix "1.1" when referring to Python version (e.g., "Python 1.1" → "Python 3.x")
    (r'\bPython\s+1\.\d\b',        'Python 3.x'),

    # Fix "100 days of code challenge. You will get 100 days of code challenge." repetitions
    (r'([\w\s]{20,}?)[.!?]\s+\1', r'\1.'),

    # Remove filler-only sentences
    (r'\bSo I will\b', 'I will'),
    (r'\bSo you will\b', 'You will'),
    (r'\bSo we will\b', 'We will'),

    # Fix "I am in Mac OS. So I have 1.1 for Mac OS" type noise
    (r'I (am|have|will use) (it|this) for my (app|computer)\.\s*', ''),
]


# ─────────────────────────────────────────────────────────────────
# Main clean function (Layer 1 + 2)
# ─────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Apply static corrections + regex fixes to a transcript chunk."""
    cleaned = text

    # Layer 1: dictionary replacement (case-insensitive)
    for wrong, right in CORRECTION_DICT.items():
        # Use word-boundary matching for accuracy
        pattern = r'\b' + re.escape(wrong) + r'\b'
        cleaned = re.sub(pattern, right, cleaned, flags=re.IGNORECASE)

    # Layer 2: regex patterns
    for pattern, replacement in REGEX_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Clean up extra whitespace
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    return cleaned


# ─────────────────────────────────────────────────────────────────
# LAYER 3: Query expansion for better retrieval
# When user asks a question, expand it with related terms
# so that even if transcript has wrong spelling, we still match
# ─────────────────────────────────────────────────────────────────
QUERY_EXPANSION: dict[str, list[str]] = {
    "string":       ["strength", "str", "string methods", "strings"],
    "list":         ["lists", "array", "list methods"],
    "dictionary":   ["dict", "dictonary", "dictionery", "key value"],
    "tuple":        ["tuples", "immutable"],
    "function":     ["funtion", "def", "functions", "methods"],
    "class":        ["clases", "classs", "object", "OOP"],
    "exception":    ["excpetion", "error handling", "try except"],
    "lambda":       ["lamdba", "lamba", "anonymous function"],
    "decorator":    ["decorater", "wrapper", "@"],
    "recursion":    ["recusion", "recursive"],
    "inheritance":  ["inheritence", "parent class", "child class"],
    "loop":         ["for loop", "while loop", "iteration"],
    "import":       ["module", "pip", "package"],
    "file":         ["file io", "open", "read", "write"],
    "generator":    ["generater", "yield"],
    "guido":        ["guida", "python creator", "inventor"],
    "ide":          ["idea", "editor", "vscode", "colab"],
}


def expand_query(query: str) -> list[str]:
    """
    Return the original query PLUS variant spellings to search.
    This catches transcription errors in the corpus even if user types correctly.
    """
    query_lower = query.lower()
    variants    = [query]

    for correct_term, wrong_variants in QUERY_EXPANSION.items():
        # If user used the correct term, also search for wrong variants
        if correct_term in query_lower:
            for v in wrong_variants:
                if v not in query_lower:
                    variants.append(query_lower.replace(correct_term, v))
        # If user typed a wrong variant, add the correct term too
        for v in wrong_variants:
            if v in query_lower:
                variants.append(query_lower.replace(v, correct_term))
                break

    # Deduplicate while preserving order
    seen = set()
    result = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            result.append(variant)
    return result


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_chunks = [
        "Guida 1 was the creator of Python. Python is a strength. It has strength methods.",
        "We will use an idea to write code. You can use Jupiter notebook.",
        "Welcome. We will get videos. We will get videos. We will get videos like this.",
        "Inheritence is a key concept in object oriented programming. The constractor method is __init__.",
    ]

    print("=" * 60)
    print("LAYER 1 + 2: Text Cleaning Results")
    print("=" * 60)
    for original in test_chunks:
        cleaned = clean_text(original)
        print("ORIGINAL:", original)
        print("CLEANED :", cleaned)
        print()

    print("=" * 60)
    print("LAYER 3: Query Expansion Results")
    print("=" * 60)
    test_queries = [
        "What is string immutability?",
        "How does recursion work?",
        "Explain exception handling",
        "What is a lambda function?",
    ]
    for q in test_queries:
        variants = expand_query(q)
        print(f"Query   : {q}")
        print(f"Variants: {variants}")
        print()
