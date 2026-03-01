"""
LangChain Retrieval Layer for RAG System

This module demonstrates how LangChain abstracts the vector database operations
we built from scratch. Where our dense.py manually handled batching, API calls,
and Qdrant client initialization, LangChain's QdrantVectorStore handles all of
that internally, letting us focus on the retrieval logic.

Key abstractions:
- OpenAIEmbeddings: Replaces our custom embedding loop with .embed_documents()
- QdrantVectorStore: Wraps Qdrant client lifecycle and search operations
- BaseRetriever: Abstracts search_type, search_kwargs, and the query flow
"""

from typing import Any, Optional, List
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def build_vectorstore(
    chunks: List[Any],
    collection_name: str,
    qdrant_url: str,
    openai_api_key: str,
    model: str = "text-embedding-3-small",
) -> QdrantVectorStore:
    """
    Build and initialize a Qdrant vectorstore from document chunks.

    This function abstracts the from-scratch workflow:
    - Our dense.py: Explicit loop over chunks → embed_batch() → upsert() calls
    - LangChain: QdrantVectorStore.from_documents() handles batching, embedding,
      and upserting in one call.

    Args:
        chunks: List of chunk objects with .text_with_heading (fallback: .text)
                and .metadata attributes (e.g., from chunk.py or pdf_parser.py)
        collection_name: Qdrant collection name (e.g., "financial_reports_v1")
        qdrant_url: Qdrant server URL (e.g., "http://localhost:6333")
        openai_api_key: OpenAI API key for embedding model
        model: Embedding model name (default: text-embedding-3-small)

    Returns:
        QdrantVectorStore: Initialized vectorstore ready for retrieval

    Example:
        >>> from chunk import extract_chunks
        >>> from pdf_parser import parse_pdfs
        >>>
        >>> docs = parse_pdfs("./data/financial_reports/")
        >>> chunks = extract_chunks(docs)
        >>> vectorstore = build_vectorstore(
        ...     chunks,
        ...     "annual_reports_2024",
        ...     "http://localhost:6333",
        ...     "sk-..."
        ... )
    """
    # Convert chunks to LangChain Document objects
    # Each chunk has text_with_heading and metadata from our chunking pipeline
    documents = []
    for chunk in chunks:
        # Try text_with_heading first (our from-scratch format),
        # fall back to text if not present
        text = getattr(chunk, "text_with_heading", None) or getattr(chunk, "text", "")
        metadata = getattr(chunk, "metadata", {})

        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    # Initialize embeddings (LangChain wraps OpenAI API calls)
    embeddings = OpenAIEmbeddings(model=model, api_key=openai_api_key)

    # Create client and ensure collection exists
    # (LangChain handles this internally in from_documents)
    client = QdrantClient(url=qdrant_url)

    # Create vectorstore from documents
    # Under the hood, LangChain batches documents, calls embeddings API,
    # and performs upserts—all the manual work we did in dense.py
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_url,
        collection_name=collection_name,
        prefer_grpc=False,
    )

    return vectorstore


def get_retriever(
    vectorstore: QdrantVectorStore,
    top_k: int = 5,
    metadata_filter: Optional[dict] = None,
) -> BaseRetriever:
    """
    Convert a QdrantVectorStore to a LangChain Retriever.

    In our from-scratch approach, we manually performed similarity_search()
    and ranked results. LangChain's as_retriever() standardizes this interface
    and enables it to be plugged into chains (see chain.py).

    Args:
        vectorstore: Initialized QdrantVectorStore
        top_k: Number of top results to retrieve (default: 5)
        metadata_filter: Optional Qdrant filter for metadata-based search
                        (e.g., {"source": "Q4_2024_Report.pdf"})

    Returns:
        BaseRetriever: LangChain retriever for use in RAG chains

    Example:
        >>> retriever = get_retriever(vectorstore, top_k=10)
        >>> relevant_docs = retriever.invoke({"query": "revenue growth"})
    """
    search_kwargs = {
        "k": top_k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    # as_retriever() wraps the vectorstore's similarity_search in a Runnable
    # This makes it compatible with LCEL chains (see build_rag_chain)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    return retriever


def load_vectorstore(
    collection_name: str,
    qdrant_url: str,
    openai_api_key: str,
    model: str = "text-embedding-3-small",
) -> QdrantVectorStore:
    """
    Load an existing Qdrant collection as a LangChain vectorstore.

    This function is useful for reusing vectorstores across sessions without
    re-embedding all documents. It demonstrates that LangChain treats vector
    storage as a simple lookup: given embeddings and a collection name,
    reconnect and continue using it.

    Args:
        collection_name: Qdrant collection to load (must already exist)
        qdrant_url: Qdrant server URL
        openai_api_key: OpenAI API key (needed for retrieval embedding)
        model: Embedding model name (must match the model used at creation)

    Returns:
        QdrantVectorStore: Loaded vectorstore

    Example:
        >>> # Session 1: Create and store
        >>> vectorstore = build_vectorstore(chunks, "reports_v1", ...)
        >>>
        >>> # Session 2: Reload without re-embedding
        >>> vectorstore = load_vectorstore("reports_v1", ...)
        >>> retriever = get_retriever(vectorstore)
    """
    embeddings = OpenAIEmbeddings(model=model, api_key=openai_api_key)

    # Reconnect to existing collection
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
        prefer_grpc=False,
    )

    return vectorstore
