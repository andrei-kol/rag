"""
Parse PDF files using PyMuPDF (fitz) for text and pdfplumber for tables.

Why two libraries?
- PyMuPDF (fitz): fast, reliable text extraction with block ordering preserved.
  Best for the main body text of a document.
- pdfplumber: slower but understands table geometry (rows, columns, cells).
  Without it, a financial table collapses into an unreadable string.

Design decisions:
- Text and tables are kept separate so downstream code can decide how to
  represent tables (e.g. as markdown, as plain text, or skip them).
- Metadata is passed in from outside — the parser does not guess client names
  or document types. That is the caller's responsibility.
- Page markers ([Page N]) are injected so chunkers can later attach page
  numbers to each chunk without re-parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


@dataclass
class ParsedPage:
    """Content extracted from a single PDF page."""

    page_number: int          # 1-based
    text: str                 # Raw text block from PyMuPDF
    tables: list[list[list[str | None]]] = field(default_factory=list)
    # tables: list of tables; each table is a list of rows; each row is a list of cells


@dataclass
class ParsedDocument:
    """Result of parsing one document."""

    pages: list[ParsedPage]
    metadata: dict
    source_file: str

    @property
    def full_text(self) -> str:
        """Concatenated text from all pages with page markers."""
        parts = []
        for page in self.pages:
            if page.text.strip():
                parts.append(f"[Page {page.page_number}]\n{page.text.strip()}")
        return "\n\n".join(parts)

    @property
    def all_tables(self) -> list[list[list[str | None]]]:
        """All tables across all pages."""
        result = []
        for page in self.pages:
            result.extend(page.tables)
        return result

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def tables_as_text(self) -> str:
        """
        Convert all tables to a readable text representation.
        Useful for embedding table content alongside regular text.
        """
        if not self.all_tables:
            return ""
        parts = []
        for i, table in enumerate(self.all_tables):
            rows = []
            for row in table:
                cleaned = [cell or "" for cell in row]
                rows.append(" | ".join(cleaned))
            parts.append(f"[Table {i + 1}]\n" + "\n".join(rows))
        return "\n\n".join(parts)


def _clean_text(text: str) -> str:
    """
    Light cleaning: collapse multiple blank lines, strip trailing whitespace.
    We intentionally keep newlines and paragraph structure intact —
    the chunker will deal with boundaries, not the parser.
    """
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def parse_pdf(
    path: str | Path,
    metadata: dict | None = None,
    extract_tables: bool = True,
) -> ParsedDocument:
    """
    Parse a PDF file into a ParsedDocument.

    Args:
        path: Path to the PDF file.
        metadata: Caller-supplied metadata (client_id, doc_type, year, etc.).
                  The parser adds source_file and page_count on top.
        extract_tables: Whether to run pdfplumber for table extraction.
                        Set to False for speed when you know there are no tables.

    Returns:
        ParsedDocument with per-page text and tables.

    Example:
        doc = parse_pdf(
            "data/raw/clients/acme_corp/financial_report_2023.pdf",
            metadata={
                "client_id": "acme_corp",
                "doc_type": "financial_report",
                "year": 2023,
                "confidentiality": "internal",
            },
        )
        print(doc.full_text[:500])
        print(f"Found {len(doc.all_tables)} tables")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    metadata = dict(metadata or {})
    metadata["source_file"] = str(path)
    metadata["filename"] = path.name

    parsed_pages: list[ParsedPage] = []

    # --- Step 1: Extract text with PyMuPDF ---
    fitz_doc = fitz.open(str(path))
    page_texts: dict[int, str] = {}  # page_index -> text

    for page_idx, page in enumerate(fitz_doc):
        # "text" layout: preserves reading order of text blocks
        raw_text = page.get_text("text")
        page_texts[page_idx] = _clean_text(raw_text)

    fitz_doc.close()

    # --- Step 2: Extract tables with pdfplumber (optional) ---
    page_tables: dict[int, list] = {i: [] for i in range(len(page_texts))}

    if extract_tables:
        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    page_tables[page_idx] = tables

    # --- Step 3: Combine into ParsedPage objects ---
    for page_idx in sorted(page_texts.keys()):
        parsed_pages.append(
            ParsedPage(
                page_number=page_idx + 1,  # convert to 1-based
                text=page_texts[page_idx],
                tables=page_tables[page_idx],
            )
        )

    metadata["page_count"] = len(parsed_pages)

    return ParsedDocument(
        pages=parsed_pages,
        metadata=metadata,
        source_file=str(path),
    )
