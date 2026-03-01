"""
Document compression and reranking via LangChain's retriever wrappers.

This module provides high-level functions to add reranking (compression) to
any retriever using LangChain's ContextualCompressionRetriever.

WHAT IS CONTEXTUAL COMPRESSION?
A retriever wrapper that post-processes results from a base retriever using
a "compressor" - a function that filters, reranks, or transforms documents.
Unlike the base retriever, the compressor has access to the original query,
allowing context-aware document selection.

RERANKING VS COMPRESSION:
- Reranking: Re-score documents, keep top-k
- Compression: Compress document length (extract relevant snippets)
- Filtering: Remove low-scoring documents

ContextualCompressionRetriever can do any of these. Common use case:
combine reranking (score all docs) with optional compression (shorten docs).
"""

from typing import List, Optional, Any
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker as LCCrossEncoderReranker
from langchain_core.schema import BaseRetriever


def build_reranking_retriever(
    base_retriever: BaseRetriever,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 3
) -> ContextualCompressionRetriever:
    """
    Wrap a base retriever with cross-encoder reranking.
    
    ARCHITECTURE:
    base_retriever (e.g., VectorStore)
         ↓ retrieve top-20
    ContextualCompressionRetriever
         ↓ rerank with cross-encoder
    LCCrossEncoderReranker
         ↓ keep top-n
    return results
    
    This is LangChain's way of doing what our from_scratch reranker.py does
    manually. The benefit: it integrates seamlessly with LangChain's ecosystem
    and handles edge cases (no documents, empty queries, etc).
    
    Args:
        base_retriever: Any LangChain retriever (VectorStore, BM25, etc).
        model_name: Cross-encoder model identifier.
        top_n: Number of reranked results to return.
    
    Returns:
        ContextualCompressionRetriever that reranks results from base_retriever.
    
    Example:
        >>> vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        >>> reranking_retriever = build_reranking_retriever(
        ...     vector_retriever,
        ...     top_n=3
        ... )
        >>> docs = reranking_retriever.get_relevant_documents("bond pricing models")
    """
    # Create the cross-encoder compressor
    compressor = LCCrossEncoderReranker(
        model=model_name,
        top_n=top_n
    )
    
    # Wrap the base retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever


def build_full_advanced_retriever(
    vectorstore,
    sparse_chunks: List,
    top_k_retrieve: int = 20,
    top_n_rerank: int = 3
) -> ContextualCompressionRetriever:
    """
    Build a full advanced retrieval pipeline: ensemble search + reranking.
    
    PIPELINE:
    Query
      ↓
    EnsembleRetriever (dense + sparse) → top-20 candidates
      ↓
    ContextualCompressionRetriever + CrossEncoderReranker → top-3 precise
      ↓
    Results
    
    This combines:
    1. Hybrid retrieval (dense + sparse) for breadth
    2. Cross-encoder reranking for precision
    
    The result: recalls documents that match semantically OR lexically,
    then reranks using a more powerful model for final precision.
    
    Args:
        vectorstore: LangChain VectorStore instance (e.g., FAISS, Chroma).
        sparse_chunks: List of chunks for BM25 retriever.
        top_k_retrieve: Number of candidates to retrieve before reranking (default 20).
        top_n_rerank: Final number of results after reranking (default 3).
    
    Returns:
        ContextualCompressionRetriever with full advanced pipeline.
    
    Example:
        >>> from langchain_community.vectorstores import FAISS
        >>> vectorstore = FAISS.from_documents(docs, embeddings)
        >>> full_retriever = build_full_advanced_retriever(
        ...     vectorstore,
        ...     chunks,
        ...     top_k_retrieve=20,
        ...     top_n_rerank=3
        ... )
        >>> results = full_retriever.get_relevant_documents("Fed monetary policy impacts")
    """
    # Import here to avoid circular dependencies
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    # Build dense retriever from vectorstore
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k_retrieve}
    )
    
    # Build sparse (BM25) retriever
    documents = []
    for chunk in sparse_chunks:
        # Extract text
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
    
    sparse_retriever = BM25Retriever.from_documents(
        documents,
        k=top_k_retrieve
    )
    
    # Combine with ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=(0.7, 0.3)  # 70% dense, 30% sparse
    )
    
    # Wrap with reranking
    compressor = LCCrossEncoderReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=top_n_rerank
    )
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return final_retriever


# RERANKING ARCHITECTURE NOTES
# =============================
# This module builds on three LangChain concepts:
#
# 1. BaseRetriever
#    Interface: .get_relevant_documents(query) -> List[Document]
#    Any retriever implements this (VectorStore, BM25, etc).
#
# 2. EnsembleRetriever
#    Combines multiple retrievers using Reciprocal Rank Fusion (RRF).
#    Calls each base retriever, fuses results by rank.
#
# 3. ContextualCompressionRetriever
#    Wraps a base retriever and applies a compressor function.
#    Compressor has access to query + documents for context-aware filtering.
#    LCCrossEncoderReranker is one type of compressor (scoring + filtering).
#
# FLOW:
# ContextualCompressionRetriever.get_relevant_documents(query)
#   → base_retriever.get_relevant_documents(query)  [get top-20]
#   → compressor.compress_documents(documents, query)  [rerank to top-3]
#   → return results
#
# The key insight: the compressor function runs AFTER retrieval, so it has
# access to both the query and all candidate documents for joint scoring.
# This is why cross-encoders are more accurate than bi-encoders.
