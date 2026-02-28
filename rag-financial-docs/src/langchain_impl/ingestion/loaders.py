"""
LangChain document loaders for PDF and DOCX files.

LangChain's loaders return a list of Document objects:
    Document(page_content="...", metadata={...})

This is the standard interface across all LangChain components — the same
object flows from loader → splitter → embedder → vector store.
Compare this to our from-scratch parsers which return ParsedDocument /
ParsedDocxDocument. Different shape, same idea.

When to use LangChain loaders vs. from-scratch:
  - PyPDFLoader: great for standard text PDFs. Fast, zero config.
  - UnstructuredPDFLoader: slower but handles complex layouts better.
  - For financial tables: our custom pdf_parser + table_parser wins —
    LangChain loaders don't give you structured table access.
  - DirectoryLoader: load an entire folder in one call — very convenient
    for the ingestion pipeline.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document


def load_pdf(
    path: str | Path,
    metadata: dict | None = None,
    use_unstructured: bool = False,
) -> list[Document]:
    """
    Load a PDF and return a list of LangChain Documents (one per page).

    Args:
        path: Path to the PDF file.
        metadata: Extra metadata merged into each Document's metadata dict.
        use_unstructured: If True, use UnstructuredPDFLoader (slower, better layout).
                          Default: PyPDFLoader (fast, good enough for most PDFs).

    Returns:
        List of Document objects. PyPDFLoader returns one Document per page;
        UnstructuredPDFLoader may return one per element (paragraph, title, etc.).

    Example:
        docs = load_pdf(
            "data/raw/clients/acme_corp/financial_report_2023.pdf",
            metadata={"client_id": "acme_corp", "doc_type": "financial_report"},
        )
        print(f"Loaded {len(docs)} page documents")
        print(docs[0].page_content[:200])
        print(docs[0].metadata)
        # {'source': '...', 'page': 0, 'client_id': 'acme_corp', 'doc_type': '...'}
    """
    path = str(path)
    loader_cls = UnstructuredPDFLoader if use_unstructured else PyPDFLoader
    loader = loader_cls(path)
    documents = loader.load()

    if metadata:
        for doc in documents:
            doc.metadata.update(metadata)

    return documents


def load_docx(
    path: str | Path,
    metadata: dict | None = None,
) -> list[Document]:
    """
    Load a DOCX file and return a list of LangChain Documents.

    Docx2txtLoader returns the entire document as a single Document object.

    Args:
        path: Path to the .docx file.
        metadata: Extra metadata merged into the Document's metadata dict.

    Returns:
        List with a single Document containing the full document text.

    Example:
        docs = load_docx(
            "data/raw/policies/risk_management.docx",
            metadata={"doc_type": "policy", "confidentiality": "internal"},
        )
        print(docs[0].page_content[:300])
    """
    loader = Docx2txtLoader(str(path))
    documents = loader.load()

    if metadata:
        for doc in documents:
            doc.metadata.update(metadata)

    return documents


def load_directory(
    directory: str | Path,
    glob: str = "**/*.pdf",
    metadata_fn: callable | None = None,
    use_unstructured: bool = False,
) -> list[Document]:
    """
    Load all matching files from a directory recursively.

    Args:
        directory: Root directory to search.
        glob: Glob pattern for file matching. Examples:
              "**/*.pdf"  — all PDFs recursively
              "**/*.{pdf,docx}"  — PDFs and DOCX (note: DirectoryLoader uses fnmatch)
        metadata_fn: Optional callable (path -> dict) to attach metadata per file.
                     If None, no extra metadata is added beyond what the loader provides.
        use_unstructured: Passed through to the PDF loader.

    Returns:
        All loaded Documents from all matched files.

    Example:
        # Load all Acme Corp documents and tag them
        docs = load_directory(
            "data/raw/clients/acme_corp",
            glob="**/*.pdf",
            metadata_fn=lambda p: {"client_id": "acme_corp"},
        )
        print(f"Loaded {len(docs)} documents from Acme Corp")
    """
    loader_cls = UnstructuredPDFLoader if use_unstructured else PyPDFLoader

    loader = DirectoryLoader(
        str(directory),
        glob=glob,
        loader_cls=loader_cls,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()

    if metadata_fn:
        for doc in documents:
            source = doc.metadata.get("source", "")
            extra = metadata_fn(Path(source))
            doc.metadata.update(extra)

    return documents
