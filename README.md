# NotebookLM-like Knowledge Base (RAG) System

Enterprise-ready, modular RAG knowledge notebook inspired by Google NotebookLM. Pluggable retrieval strategies, configurable multi-LLM providers, scalable vector DBs (Chroma/Milvus), and a minimal React UI.

## Key Capabilities

- Modular RAG pipeline:
  - Ingestion (local file path or raw text), chunking, embeddings
  - Hybrid retrieval (BM25 + Vector) with optional reranking
- Pluggable vector stores:
  - Chroma (local persistence, no external infra)
  - Milvus (via docker-compose infra profile)
- Configurable models:
  - OpenAI-compatible LLM endpoints (OpenAI, DeepSeek, Doubao/Ark, custom/vLLM/TGI)
  - Embeddings: OpenAI or custom-compatible
- UI:
  - Simple Notebook + Source + Chat workflow with citations
- Deployments:
  - Docker-first
  - Local dev (Python + Node)

## Architecture

React (Vite) → REST API (FastAPI) → RAG Pipeline  
- Ingest → Chunk → Embed → Vector Store (Chroma/Milvus)  
- Retrieve (BM25 + Vector) → Optional Rerank → LLM Generate → Citations

Optional Infra (via compose infra profile): Milvus, Neo4j

## Quick Start

### Option A: Docker (Lite, Chroma; no external infra)

Prereqs: Docker Desktop (Compose v2)

1) Copy env and configure
```bash
cp .env.example .env
# In .env set:
# VECTOR_STORE=chroma
# LLM_API_KEY=your-key
# Optionally adjust LLM_BASE_URL/LLM_MODEL
```

2) Start backend (lite profile)
```bash
docker compose --profile lite up -d --build
```

3) Verify
- Open http://localhost:8000/health

4) Frontend (dev, run on your machine)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Option B: Local Development (no Docker)

Prereqs:
- Python 3.11+ (recommended)
- Node.js v20.x

1) Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Configure env: create .env-local or .env with VECTOR_STORE=chroma and LLM_API_KEY
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Verify: http://localhost:8000/health
```

2) Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## API Endpoints

- GET /health
- GET /api/settings
- POST /api/settings/provider { "provider": "openai|deepseek|gemini|doubao" }
- POST /api/notebooks { "name": "Demo" }
- GET /api/notebooks
- POST /api/notebooks/{id}/upload { "file_path": "/abs/or/relative/path", "metadata": {} }
- POST /api/notebooks/{id}/uploadText { "texts": ["..."], "metadata": {} }
- POST /api/notebooks/{id}/query { "question": "?", "top_k": 5 }

## License

MIT