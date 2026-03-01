# 03: Embeddings and RAG Pipeline - Comprehensive Notebook

## File Location
`/sessions/clever-magical-ride/mnt/rag/rag-financial-docs/notebooks/03_embeddings_and_rag.ipynb`

## Overview
A complete educational Jupyter notebook teaching beginners how embeddings work and how to build an end-to-end RAG (Retrieval-Augmented Generation) pipeline for financial documents.

## Notebook Structure (20 cells total)

### Section 1: Setup & Imports
- **Cell 1 (Markdown)**: Title and course context
- **Cell 2 (Markdown)**: Architecture overview with ASCII diagram showing:
  - Question → Embed → [Qdrant Search] → Top-K Chunks → GPT-4o-mini → Answer
  - Offline indexing phase (chunks → embed → index)
- **Cell 3 (Code)**: Complete import setup including:
  - Standard library (os, sys, pathlib, json, datetime)
  - Data processing (numpy, tqdm)
  - LLM/Embeddings (openai, langchain_openai)
  - Vector database (qdrant_client)
  - Configuration (dotenv)

### Section 2: What is an Embedding?
- **Cell 4 (Markdown)**: 
  - Embeddings as semantic coordinates in meaning-space
  - Similar meaning = nearby points concept
  - text-embedding-3-small produces 1536 dimensions
  - Why cosine similarity > Euclidean distance
  - Visual intuition diagram
- **Cell 5 (Code)**: 
  - Conceptual demo (no API needed)
  - Simulates embeddings with fake vectors
  - Demonstrates cosine similarity calculations
  - Shows similar vs. different topics

### Section 3: OpenAI Embeddings
- **Cell 6 (Markdown)**:
  - text-embedding-3-small specs and pricing ($0.02/1M tokens)
  - Comparison table with other models (text-embedding-3-large, ada-002)
  - Cost calculations (1000 chunks example = $0.002)
  - Batching best practices
- **Cell 7 (Code)**:
  - `embed_single_text()` function
  - `embed_batch()` with batching support (up to 2048 per request)
  - `estimate_embedding_cost()` calculator
  - Cost examples for different chunk counts

### Section 4: Qdrant — Vector Database
- **Cell 8 (Markdown)**:
  - Why vector databases (naive list search vs. HNSW indexing)
  - Qdrant concepts: Collections, Points, Payloads, Distance metrics
  - Docker setup instructions
  - In-memory option for testing
- **Cell 9 (Code)**:
  - `create_qdrant_collection()` function
  - `insert_points_to_qdrant()` function
  - Example with 1536-dimensional fake vectors
  - Mock payload structure with metadata

### Section 5: DenseRetriever (From Scratch)
- **Cell 10 (Markdown)**:
  - DenseRetriever class design
  - index_chunks() and search() methods
  - Text-with-heading trick for semantic context
  - Payload design for storing original text
- **Cell 11 (Code)**:
  - `MockDenseRetriever` class implementation showing:
    - `index_chunks()`: embed + batch + store in Qdrant
    - `search()`: embed question + vector search
  - Both methods fully documented
- **Cell 12 (Code)**:
  - Sample financial document (Annual Report 2023)
  - Document chunking demo
  - Creates mock chunks with metadata
- **Cell 13 (Code)**:
  - Retriever usage example
  - API key guard (graceful degradation if no key)
  - Shows expected output format

### Section 6: Generating Answers (From Scratch)
- **Cell 14 (Markdown)**:
  - Answer generation pipeline (4 steps)
  - Prompt design best practices
  - Numbered citations [1], [2], etc.
  - Temperature=0 for factual answers
  - Explicit "I don't know" instructions to prevent hallucinations
  - Example system prompt
- **Cell 15 (Code)**:
  - SYSTEM_PROMPT definition
  - `build_context_block()` function with numbered citations
  - `build_prompt()` function for full prompt construction
  - Demo with mock chunks showing formatted output
- **Cell 16 (Code)**:
  - `RAGGenerator` class with `generate()` method
  - Token counting and cost calculation
  - gpt-4o-mini pricing ($0.15/$0.60 per 1M tokens)
  - API key guard with expected output

### Section 7: LangChain LCEL Pipeline
- **Cell 17 (Markdown)**:
  - Why LangChain (declarative pipelines, streaming, tracing)
  - Code comparison: from scratch vs. LangChain
  - Feature comparison table
  - Reduced complexity (~50 lines → ~10 lines)
- **Cell 18 (Code)**:
  - `build_simple_rag_chain()` function
  - LangChain ChatPromptTemplate
  - LCEL chain composition with pipe operator
  - Complete ~15 line example

### Section 8: Comparison & Future (From Scratch vs LangChain)
- **Cell 19 (Markdown)**:
  - Comprehensive comparison table:
    - Lines of code, Setup time, Learning value
    - Customization, Streaming, Debugging
    - Production readiness, Community
  - Recommendations for different use cases
  - Week 4 preview (hybrid search, reranking, metadata filtering)
  - Architecture evolution diagram
- **Cell 20 (Markdown)**:
  - Summary of key concepts learned
  - Key formulas (cosine similarity, cost estimation, RAG pipeline)
  - Next steps (setup, experimentation, UI building)

## Key Features

### Educational Value
- Progresses from theory (embeddings) to practice (full RAG)
- Code examples that run without API keys (graceful degradation)
- Clear explanations with visual ASCII diagrams
- Comparison of multiple implementation approaches

### Practical Code Examples
- All functions are fully documented with docstrings
- Type hints throughout
- Production-ready patterns (batching, error handling)
- Cost estimation helpers
- Real financial document examples

### Coverage
- **Theory**: Embeddings, cosine similarity, semantic search
- **APIs**: OpenAI embeddings, chat completions
- **Databases**: Qdrant setup, point insertion, vector search
- **Engineering**: Batching, cost optimization, metadata
- **Frameworks**: LangChain LCEL, component composition
- **Best Practices**: Prompt design, citations, hallucination prevention

### No External Dependencies Required
- Code designed to run with standard imports
- API calls guarded by environment variable checks
- Mock/demo data included for offline learning
- Expected outputs shown when API unavailable

## File Statistics
- **Size**: 45KB
- **Lines**: 1058
- **Cells**: 20 (10 markdown, 10 code)
- **Valid JSON**: Yes (nbformat 4.5)

## Usage

1. **View the notebook**: Open in Jupyter Lab/Notebook
   ```bash
   jupyter lab 03_embeddings_and_rag.ipynb
   ```

2. **Prerequisites**: 
   - Python 3.9+
   - Jupyter
   - OpenAI API key (optional, gracefully handled)
   - Local Qdrant instance (optional)

3. **Run cells**: Execute from top to bottom or selectively

4. **Modify and Experiment**: All code is well-commented and modular

## Next Steps Suggested in Notebook
1. Set up Qdrant locally with Docker
2. Create notebooks with own financial documents
3. Experiment with chunk sizes and top-K values
4. Build web UI with Streamlit/Gradio
5. Explore Week 4 advanced techniques

## Integration Points
This notebook fits into a larger educational series:
- **Week 1-2**: Document parsing and chunking (earlier notebooks)
- **Week 3**: This notebook (embeddings and RAG fundamentals)
- **Week 4**: Advanced retrieval (hybrid search, reranking, filtering)
- **Week 5+**: Production systems, evaluation, deployment

---

Created: 2026-02-28
Course: RAG for Financial Documents
Level: Beginner to Intermediate
