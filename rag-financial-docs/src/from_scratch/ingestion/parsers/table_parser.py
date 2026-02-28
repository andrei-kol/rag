"""
Utilities for normalising and representing tables extracted from PDFs and DOCX files.

Tables in financial documents are critical — they contain the actual numbers.
This module provides helpers to:
  1. Clean raw table data (strip whitespace, handle None cells)
  2. Convert tables to markdown format (for better LLM comprehension)
  3. Wrap tables in a TableChunk with contextual metadata so they can be
     stored in the vector DB alongside regular text chunks.

Why store tables differently?
- A table cell like "1,234,567" means nothing without the row/column headers.
- We need to inject that context when embedding, otherwise retrieval fails.
- Solution: represent each table as a self-contained text block with headers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TableChunk:
    """
    A table converted to text, ready to be embedded and stored.

    Example text_representation for a financial table:

        [Table: Revenue Breakdown]
        | Quarter | Revenue ($M) | YoY Change |
        |---------|-------------|------------|
        | Q1 2023 | 12.4        | +8%        |
        | Q2 2023 | 13.1        | +11%       |
    """

    text_representation: str          # The embeddable text version of the table
    raw_table: list[list[str]]        # Original rows/cells
    metadata: dict = field(default_factory=dict)


def clean_table(
    table: list[list[str | None]],
) -> list[list[str]]:
    """
    Strip whitespace and replace None cells with empty strings.

    Args:
        table: Raw table from pdf_parser or docx_parser (may contain None).

    Returns:
        Cleaned table where every cell is a non-None string.
    """
    cleaned = []
    for row in table:
        cleaned_row = [(cell.strip() if cell else "") for cell in row]
        cleaned.append(cleaned_row)
    return cleaned


def table_to_markdown(
    table: list[list[str]],
    title: str = "",
) -> str:
    """
    Convert a table (list of rows) to a GitHub-flavoured markdown table.

    The first row is treated as the header.

    Args:
        table: Cleaned table (no None values).
        title: Optional title prepended to the markdown block.

    Returns:
        Markdown string.

    Example:
        >>> rows = [["Item", "Q1", "Q2"], ["Revenue", "12.4", "13.1"]]
        >>> print(table_to_markdown(rows, title="Revenue"))
        [Table: Revenue]
        | Item    | Q1   | Q2   |
        |---------|------|------|
        | Revenue | 12.4 | 13.1 |
    """
    if not table:
        return ""

    lines = []
    if title:
        lines.append(f"[Table: {title}]")

    header = table[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in table[1:]:
        # Pad short rows to match header length
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def make_table_chunk(
    raw_table: list[list[str | None]],
    metadata: dict | None = None,
    title: str = "",
) -> TableChunk:
    """
    Full pipeline: clean a raw table and wrap it in a TableChunk.

    Args:
        raw_table: As returned by pdf_parser or docx_parser.
        metadata: Metadata dict (will be copied, not mutated).
        title: Human-readable title for the table (e.g. "Revenue Breakdown").

    Returns:
        TableChunk ready for embedding.

    Example:
        chunk = make_table_chunk(
            raw_table=doc.all_tables[0],
            metadata={"client_id": "acme_corp", "doc_type": "financial_report"},
            title="Annual Revenue",
        )
        # chunk.text_representation  -> embeddable markdown
        # chunk.metadata             -> {client_id, doc_type, content_type: "table"}
    """
    cleaned = clean_table(raw_table)
    md = table_to_markdown(cleaned, title=title)

    chunk_metadata = dict(metadata or {})
    chunk_metadata["content_type"] = "table"
    if title:
        chunk_metadata["table_title"] = title

    return TableChunk(
        text_representation=md,
        raw_table=cleaned,
        metadata=chunk_metadata,
    )
