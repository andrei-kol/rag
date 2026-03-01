"""
Ensemble retrieval combining dense and sparse search via LangChain.

This module provides functions to build hybrid retrieval systems that combine
dense vector search (semantic similarity) with sparse search (exact term matching).
LangChain's EnsembleRetriever handles the combination logic transparently.

DENSE VS SPARSE SEARCH:
- Dense (semantic): "What is financial leverage?" matches "debt-to-equity ratio"
- Sparse (lexical): Matches exact keywords only, misses synonyms

Ensemble retrieval: Run both, combine results using Reciprocal Rank Fusion (RRF)
This ensures both semantic and exact-match relevance are captured.
"""

from typing import List, Optional, Tuple
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document


def build_bm25_retriever(
    chunks: List,
    k: int = 5
) -> BM25Retriever:
    """
    Build a BM25 (sparse/lexical) retriever from document chunks.
    
    BM25 is a ranking function that scores documents based on keyword frequency
    and document length. It's fast, effective for exact-match queries, and
    requires no training. It's the baseline for sparse retrieval.
    
    Args:
        chunks: List of document chunks. Each chunk should have:
                - .text or .text_with_heading attribute (preferred)
                - Be convertible to str() if no text attribute
        k: Number of results to retrieve (default 5).
    
    Returns:
        BM25Retriever instance ready for .get_relevant_documents(query) calls.
    
    Example:
        >>> chunks = [Chunk(text="bond yields..."), ...]
        >>> bm25_retriever = build_bm25_retriever(chunks, k=5)
        >>> docs = bm25_retriever.get_relevant_documents("bond yields")
    """
    # Convert chunks to LangChain Document format
    documents = []
    for chunk in chunks:
        # Extract text using same logic as reranker
        if hasattr(chunk, 'text_with_heading'):
            text = chunk.text_with_heading
        elif hasattr(chunk, 'text'):
            text = chunk.text
        else:
            text = str(chunk)
        
        # Extract metadata
        metadata = {}
        if hasattr(chunk, 'metadata'):
            metadata = chunk.metadata
        elif hasattr(chunk, 'source'):
            metadata = {'source': chunk.source}
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    # Create and return BM25Retriever
    return BM25Retriever.from_documents(documents, k=k)


def build_ensemble_retriever(
    dense_retriever,
    sparse_chunks: List,
    weights: Tuple[float, float] = (0.7, 0.3),
    k: int = 5
) -> EnsembleRetriever:
    """
    Build an ensemble retriever combining dense and sparse search.
    
    ENSEMBLE STRATEGY:
    Uses Reciprocal Rank Fusion (RRF) to combine two ranking lists:
    1. Dense retriever results (semantic similarity via embeddings)
    2. Sparse retriever results (exact keyword matching via BM25)
    
    RRF formula: score = sum(1 / (rank + 60)) for all lists
    
    The weights parameter controls the influence of each retriever:
    - weights=(0.7, 0.3): 70% dense, 30% sparse (more semantic)
    - weights=(0.5, 0.5): Equal influence (balanced)
    - weights=(0.3, 0.7): 30% dense, 70% sparse (more keyword-based)
    
    Args:
        dense_retriever: A LangChain retriever using embeddings (e.g., VectorStoreRetriever).
        sparse_chunks: List of chunks for BM25 retriever creation.
        weights: Tuple of (dense_weight, sparse_weight). Must sum to 1.0.
        k: Number of results to retrieve.
    
    Returns:
        EnsembleRetriever instance combining dense and sparse results.
    
    Example:
        >>> dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        >>> ensemble = build_ensemble_retriever(
        ...     dense_retriever,
        ...     chunks,
        ...     weights=(0.7, 0.3),
        ...     k=10
        ... )
        >>> results = ensemble.get_relevant_documents("financial risk factors")
    """
    # Create sparse (BM25) retriever
    sparse_retriever = build_bm25_retriever(sparse_chunks, k=k)
    
    # Combine with EnsembleRetriever
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=weights
    )
    
    return ensemble


# NOTE: Implementation comparison
# ============================
# LangChain's EnsembleRetriever uses Reciprocal Rank Fusion (RRF) internally,
# the same logic we implement in our from_scratch HybridRetriever class.
#
# DIFFERENCES:
# 1. Our HybridRetriever:
#    - Explicitly logs dense_rank and sparse_rank for each result
#    - Provides direct access to the RRF scores
#    - Good for debugging and understanding rank changes
#
# 2. LangChain's EnsembleRetriever:
#    - Cleaner API, more opaque internals
#    - Production-ready, well-tested
#    - Preferred for deployment (less code to maintain)
#
# RECOMMENDATION:
# Use LangChain's EnsembleRetriever for production systems.
# Use our from_scratch HybridRetriever for educational/debugging purposes.
