"""
LangChain text splitters for all three chunking strategies.

LangChain's splitters accept list[Document] (from loaders) and return
list[Document] -- the same interface flows through the whole pipeline.

Compare to our from-scratch chunkers which return custom dataclasses
(TextChunk, SemanticChunk, DocumentChunk). LangChain uses Document
everywhere for consistency.

Strategy            | From Scratch              | LangChain
--------------------|--------------------------|---------------------------
Fixed-size chars    | split_fixed_size()        | RecursiveCharacterTextSplitter
Semantic            | split_semantic()          | SemanticChunker (experimental)
Document-aware      | split_document_aware()    | MarkdownHeaderTextSplitter
"""

from __future__ import annotations

from langchain_core.documents import Document


def split_fixed_langchain(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Fixed-size character chunking via LangChain.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks,
    then sentence breaks, then word breaks, then characters -- in that
    order. This makes it smarter than naive character splitting while
    still being size-bounded.

    Args:
        documents: List of LangChain Document objects (from loaders).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap in characters.

    Returns:
        List of Document chunks. Each chunk inherits the source
        Document's metadata plus a 'chunk_index' field.

    Example:
        from src.langchain_impl.ingestion.loaders import load_pdf
        docs = load_pdf("data/raw/clients/acme_corp/financial_report_2023.pdf")
        chunks = split_fixed_langchain(docs, chunk_size=512, chunk_overlap=64)
        print(f"{len(chunks)} chunks from {len(docs)} pages")
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,   # adds 'start_index' to metadata
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def split_semantic_langchain(
    documents: list[Document],
    embeddings=None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> list[Document]:
    """
    Semantic chunking via LangChain's experimental SemanticChunker.

    SemanticChunker embeds each sentence and splits where the similarity
    between adjacent sentences drops sharply (configurable threshold).

    Args:
        documents: List of LangChain Documents.
        embeddings: LangChain embeddings object. Defaults to
                    OpenAIEmbeddings with text-embedding-3-small.
        breakpoint_threshold_type: How to determine split points.
                    'percentile' (default) -- split at the Nth percentile
                    of similarity drops. 'standard_deviation' and
                    'interquartile' are also available.
        breakpoint_threshold_amount: Parameter for the threshold type.
                    For 'percentile': split at this percentile (e.g. 95
                    means split where similarity drops are in the top 5%).

    Returns:
        List of Document chunks with semantic boundaries.

    Example:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        chunks = split_semantic_langchain(docs, embeddings=embeddings)
        print(f"{len(chunks)} semantic chunks")
    """
    from langchain_experimental.text_splitter import SemanticChunker

    if embeddings is None:
        from langchain_openai import OpenAIEmbeddings
        from src.config import settings
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def split_document_aware_langchain(
    documents: list[Document],
    headers_to_split_on: list[tuple[str, str]] | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Document-aware chunking via LangChain's MarkdownHeaderTextSplitter.

    First splits on markdown headers to preserve section structure,
    then applies RecursiveCharacterTextSplitter to any section that
    still exceeds chunk_size.

    Args:
        documents: List of LangChain Documents. Best used with documents
                   produced by our docx_parser (which outputs # headings).
        headers_to_split_on: List of (marker, metadata_key) pairs.
                   Defaults to H1, H2, H3.
        chunk_size: Max chars for the secondary fixed-size pass.
        chunk_overlap: Overlap for the secondary pass.

    Returns:
        List of Documents where each chunk's metadata includes the
        section headers as fields (e.g. {"Header 1": "Revenue", ...}).

    Example:
        chunks = split_document_aware_langchain(
            docs,
            headers_to_split_on=[("#", "section"), ("##", "subsection")],
        )
        for c in chunks[:3]:
            print(c.metadata.get("section"), c.page_content[:80])
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,     # keep header text in the chunk body
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []
    for doc in documents:
        # Step 1: split on headers
        header_chunks = md_splitter.split_text(doc.page_content)
        # Re-attach source document's metadata to each header chunk
        for hc in header_chunks:
            hc.metadata = {**doc.metadata, **hc.metadata}
        # Step 2: further split long sections
        final_chunks = char_splitter.split_documents(header_chunks)
        all_chunks.extend(final_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i

    return all_chunks
