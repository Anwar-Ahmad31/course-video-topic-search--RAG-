# PyTutor AI — Autonomous Video Knowledge Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BGE--M3%20(1024--dim)-orange.svg)](https://huggingface.co/BAAI/bge-m3)
[![LLM](https://img.shields.io/badge/Inference-Groq%20Cloud%20%7C%20Ollama-purple.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

PyTutor AI is a production-grade **Retrieval-Augmented Generation (RAG)** platform designed to index, search, and synthesize technical lecture video courses. It provides accurate, context-grounded technical explanations with exact citation timestamps, bridging video archives with natural language search.

---

## Architecture Overview

```
 [ Video Lectures (MP4) ]
            │
      OpenAI Whisper
            ▼
  [ Speech Transcripts ]
            │
   Phonetic Normalization (3-Layer Pipeline)
            │
   BGE-M3 Dense Embeddings (1024-dim)
            │
            ▼
  [ Vector Knowledge Base ] ◄──── User Inquiry
            │                          │
   Normalized Cosine Search            ▼
            │               Domain Query Expansion
            ▼
 [ Relevant Context Chunks ]
            │
       Llama 3.3 (Groq Cloud API / Ollama)
            │
            ▼
 [ Grounded Answer with Video Citations & Timestamps ]
```

---

## Core Capabilities

- **Zero-Hallucination Grounded Answers:** Strictly constrains response synthesis to indexed transcripts, eliminating out-of-context assertions.
- **ASR Phonetic Normalization:** Integrated 3-layer rule engine designed to correct spoken lecture ambiguities and automatic speech recognition (ASR) drift before embedding.
- **Sub-Second Low Latency:** Powered by the Groq LPU inference engine (`llama-3.3-70b-versatile`) for instant technical generation.
- **Bi-directional Query Expansion:** Expands search vectors across technical synonyms and phonetic alternatives to maximize retrieval recall.
- **Module & Timestamp Traceability:** Every response is paired with exact video module indices and start/end playback intervals.
- **Enterprise Web Interface:** Responsive dark-mode single-page application (SPA) featuring real-time telemetry, adjustable search parameters, and syntax-highlighted code rendering.

---

## Repository Structure

```
├── backend/
│   ├── main.py              # FastAPI application server and REST endpoints
│   ├── rag_engine.py        # Semantic vector search and LLM orchestrator
│   └── text_cleaner.py      # Transcript sanitization and normalization rules
├── frontend/
│   ├── index.html           # Enterprise Single Page Application (SPA)
│   └── streamlit_app.py     # Alternative prototyping frontend
├── main_embedded_jsons/     # Pre-computed 1024-dimensional vector knowledge store
├── .env.example             # Template for configuration parameters
├── DEPLOYMENT_GUIDE.md      # Detailed cloud deployment walkthrough
├── Procfile                 # Production web process definition (Render / Railway)
├── render.yaml              # Infrastructure-as-code deployment specification
├── requirements.txt         # Production Python dependencies
└── README.md                # Project documentation
```

---

## Quickstart (Local Development)

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation
Clone the repository and install the production dependencies:

```bash
git clone https://github.com/your-username/python-rag-tutor.git
cd python-rag-tutor
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the sample environment file and provide your Groq API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

*(Note: If you do not supply a Groq key, the system automatically falls back to your local Ollama daemon at `http://localhost:11434`)*

### 4. Run Application Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Client Application:** `http://localhost:8000/`
- **Interactive OpenAPI Documentation:** `http://localhost:8000/docs`

---

## REST API Specification

### `POST /ask`
Executes vector retrieval over the course knowledge base and returns a synthesized response.

**Request:**
```json
{
  "question": "What is string immutability in Python?",
  "top_k": 4
}
```

**Response:**
```json
{
  "answer": "String immutability means that once a string object is instantiated, individual characters cannot be mutated in-place...",
  "sources": [
    {
      "video_number": 11,
      "title": "Strings in Python",
      "start": 45.0,
      "end": 92.0,
      "score": 0.782,
      "text": "Strings in Python are immutable. If you attempt to modify..."
    }
  ]
}
```

### `GET /corpus/stats`
Returns telemetry regarding indexed modules and chunk densities.

---

## Cloud Deployment

This repository is pre-configured for one-click deployment on **Render**, **Railway**, or **Hugging Face Spaces**.

Refer to [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md) for full cloud configuration steps.

---

## License

This project is distributed under the terms of the [MIT License](./LICENSE).
