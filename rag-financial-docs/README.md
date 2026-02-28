# RAG for Financial & Legal Documents

> **Philosophy: "Understand First, Then Use the Right Tool"**
>
> Every feature in this project is built twice:
> 1. **From Scratch** (pure Python) — to understand what's actually happening under the hood
> 2. **LangChain** — to show the production-ready way of doing the same thing
>
> *"I showed you how it works from scratch so you understand the mechanics. In a real project, I recommend LangChain — it saves time, is well-maintained, and has a great ecosystem. But when something breaks — and it will — you now know where to look."*

---

## Business Case

A **B2B fintech/consulting platform** that serves corporate clients. When a new client is onboarded, the system accumulates various documents — financial reports, contracts, compliance policies, product documentation, and regulatory references. Employees and clients need to quickly find answers across this entire knowledge base.

**The AI Assistant answers questions like:**
- *"What was Client X's revenue in Q3 2023?"*
- *"Does this contract violate our risk exposure policy?"*
- *"What documents are required to onboard a client from Germany?"*
- *"Summarize the key terms of the NDA with Company Y"*

**Why this use-case is compelling:**
- Multiple document types (PDF, DOCX, tables, structured & unstructured)
- Accuracy is critical — financial data and legal terms, no room for hallucinations
- Requires metadata filtering (by client, document type, date, confidentiality level)
- Natural progression from simple to complex retrieval
- Directly relevant to enterprise AI engineering roles

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Industry standard for ML/AI |
| LLM | OpenAI GPT-4o-mini + GPT-4o | Cost-effective baseline + quality comparison |
| Embeddings | OpenAI text-embedding-3-small + sentence-transformers | Compare proprietary vs open-source |
| Vector DB | Qdrant (Docker) | Production-grade, supports hybrid search natively |
| Sparse Search | BM25 via rank_bm25 | For hybrid retrieval |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Lightweight, effective |
| PDF Parsing | PyMuPDF (fitz) + pdfplumber | Best combo for mixed content |
| Framework | LangChain + Plain Python | Each feature built from scratch, then LangChain equivalent |
| Evaluation | RAGAS + custom metrics | Industry standard for RAG eval |
| Observability | LangSmith | Free tier, great for debugging chains |
| UI | Streamlit | Fast prototype, good enough for demo |
| Containerization | Docker + docker-compose | Reproducibility |

---

## From Scratch vs LangChain — Summary

| Feature | From Scratch (lines) | LangChain (lines) | Recommendation |
|---|---|---|---|
| Document loading | ~80 per parser | ~15 total | ✅ LangChain (unless custom parsing needed) |
| Chunking | ~50 per strategy | ~10 per strategy | ⚖️ Both |
| Naive RAG pipeline | ~120 | ~20 | ✅ LangChain for prototyping |
| Hybrid search + reranking | ~200 | ~40 | ✅ LangChain (EnsembleRetriever) |
| Evaluation | ~100 | ~15 (RAGAS) | ⚖️ Both |
| Observability | ~150 | ~5 (LangSmith) | ✅ LangChain / LangSmith |
| Guardrails | ~80 | ~30 | ⚖️ Both |
| Agentic RAG | ~300+ | ~60 (LangGraph) | ✅ LangGraph strongly recommended |

---

## Article Series

| # | Title | Key Question | Notebook |
|---|---|---|---|
| 1 | У меня есть документы. С чего начать? | Can't we just dump everything into a folder? | `01_data_collection.ipynb` |
| 2 | Как разбить документы на куски? | Just split every 500 characters, right? | `02_chunking_strategies.ipynb` |
| 3 | Как искать по этим кускам? | Can't we just use regular text search (Ctrl+F)? | `03_naive_rag.ipynb` |
| 4 | Почему RAG иногда отвечает не то? | I gave it all the documents — why does it still hallucinate? | `04_advanced_retrieval.ipynb` |
| 5 | Как понять, хорошо ли работает мой RAG? | Just test it manually with a few questions? | `05_evaluation.ipynb` |
| 6 | Как это выглядит в продакшене? | Just deploy and forget? | `06_production.ipynb` |
| 7 | Что дальше? Agentic RAG | What if the question is too complex for a single retrieval? | `07_agentic_rag.ipynb` |

Each notebook follows this pattern:
- **Part A** — From Scratch (pure Python, step by step)
- **Part B** — LangChain (same result, fewer lines)
- **Part C** — Comparison & when to use what

---

## Project Structure

```
rag-financial-docs/
│
├── README.md
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── Makefile
│
├── docs/
│   ├── architecture.md
│   ├── decisions.md
│   └── images/
│
├── notebooks/                        # One per article
│   ├── 01_data_collection.ipynb
│   ├── 02_chunking_strategies.ipynb
│   ├── 03_naive_rag.ipynb
│   ├── 04_advanced_retrieval.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_production.ipynb
│   └── 07_agentic_rag.ipynb
│
├── src/
│   ├── config.py
│   ├── from_scratch/                 # Pure Python implementations
│   │   ├── ingestion/
│   │   │   ├── parsers/              # pdf_parser, docx_parser, table_parser
│   │   │   ├── chunking/             # fixed_size, semantic, document_aware
│   │   │   └── pipeline.py
│   │   ├── retrieval/                # dense, sparse, hybrid, reranker, metadata_filter
│   │   ├── generation/               # prompts, generator, guardrails
│   │   └── evaluation/              # dataset, metrics, benchmark
│   │
│   ├── langchain_impl/               # LangChain implementations (mirror of above)
│   │   ├── ingestion/                # loaders, splitters, pipeline
│   │   ├── retrieval/                # retriever, ensemble, compression
│   │   ├── generation/               # chain, guardrails
│   │   └── evaluation/              # ragas_eval
│   │
│   └── app/
│       └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── clients/
│   │   │   ├── acme_corp/
│   │   │   ├── globex_inc/
│   │   │   └── initech_llc/
│   │   ├── policies/
│   │   ├── product_docs/
│   │   └── regulatory/
│   ├── processed/
│   └── eval/
│       ├── questions.json
│       └── ground_truth.json
│
├── tests/
│   ├── test_parsers.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
└── scripts/
    ├── generate_synthetic_data.py
    ├── ingest.py
    ├── evaluate.py
    └── demo.py
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/rag-financial-docs.git
cd rag-financial-docs

# 2. Copy env file and add your API keys
cp .env.example .env

# 3. Start Qdrant
docker-compose up -d qdrant

# 4. Install dependencies
pip install -e ".[dev]"

# 5. Generate synthetic data
python scripts/generate_synthetic_data.py

# 6. Run ingestion pipeline
python scripts/ingest.py

# 7. Launch demo UI
streamlit run src/app/streamlit_app.py

# Or run a quick demo in the terminal
python scripts/demo.py
```

---

## Development

```bash
make setup      # Install all dependencies
make test       # Run test suite
make ingest     # Run ingestion pipeline
make evaluate   # Run evaluation suite
make demo       # Launch Streamlit UI
make qdrant     # Start Qdrant via Docker
```

---

## Synthetic Data

The project uses **three fictional client companies** with realistic documents:

- **Acme Corp** — mid-size tech company (financial reports, contracts, invoices, KYC)
- **Globex Inc** — large manufacturing company (financial reports, NDA, MSA, due diligence)
- **Initech LLC** — small startup (pitch deck, term sheet, financial projections)

Plus internal documents: Risk Management Policy, KYC/AML Procedures, Data Handling Policy, Fee Schedule, Client Onboarding Checklist.

All documents are generated with GPT-4 and formatted as proper PDFs — this process itself is covered in Article 1.

---

*Built as a learning project + article series. Each article pairs with a notebook in `/notebooks/`. Follow along on LinkedIn.*
