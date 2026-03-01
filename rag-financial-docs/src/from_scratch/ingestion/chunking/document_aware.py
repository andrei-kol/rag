"""
Document-aware chunking: respect headings, sections, and paragraphs.

Instead of splitting at character positions or semantic boundaries,
this strategy uses the document's own structure as chunk boundaries.
Headings (# Heading, ## Heading) and double newlines are the primary
split points.

Why this matters for financial/legal documents:
  - A financial report has clear sections: "Revenue", "Operating Expenses",
    "Risk Factors". Each section should be a chunk (or a small group of chunks).
  - Splitting mid-section mixes context from different topics.
  - The section heading itself is crucial metadata -- it tells the retriever
    WHAT the numbers in the chunk are about.

Algorithm:
  1. Split text on heading markers (# , ## , ### ) and double newlines.
  2. Attach the nearest parent heading to each chunk as metadata.
  3. If a section is too long, fall back to fixed-size splitting within it.
  4. If a section is too short, merge it with the next sibling.

When to use:
  - Documents parsed with our docx_parser (headings preserved as markdown)
  - PDFs where sections are clearly delimited
  - Policy documents, contracts, reports with explicit structure
  Not ideal for: unstructured prose with no headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.from_scratch.ingestion.chunking.fixed_size import TextChunk, split_fixed_size


# Regex that matches a line starting with one or more # characters (markdown heading)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class DocumentChunk:
    """A chunk that carries its section heading as context."""

    text: str
    chunk_index: int
    heading: str          # nearest parent heading (e.g. "## Revenue Breakdown")
    heading_level: int    # 1 = #, 2 = ##, etc. (0 = no heading found)
    metadata: dict = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def text_with_heading(self) -> str:
        """
        Full embeddable text: heading prepended to body.

        This is what you should embed and store. Without the heading,
        a chunk like "It increased by 18% year-over-year" is meaningless.
        With it: "## Revenue\nIt increased by 18% year-over-year."
        """
        if self.heading:
            return f"{self.heading}\n\n{self.text}"
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"DocumentChunk(index={self.chunk_index}, "
            f"heading={self.heading!r}, "
            f"text={preview!r}...)"
        )


def _extract_sections(text: str) -> list[tuple[str, str, int]]:
    """
    Split text into (heading, body, heading_level) tuples.

    Splits on markdown heading lines. Text before the first heading
    is returned with an empty heading string.

    Returns list of (heading_text, body_text, level) tuples.
    """
    sections: list[tuple[str, str, int]] = []
    last_end = 0
    last_heading = ""
    last_level = 0

    for match in _HEADING_RE.finditer(text):
        # Save body between previous heading and this one
        body = text[last_end:match.start()].strip()
        if body or last_heading:
            sections.append((last_heading, body, last_level))

        hashes = match.group(1)
        last_heading = match.group(0).strip()   # full heading line e.g. "## Revenue"
        last_level = len(hashes)
        last_end = match.end()

    # Trailing section after last heading
    body = text[last_end:].strip()
    if body or last_heading:
        sections.append((last_heading, body, last_level))

    return sections


def split_document_aware(
    text: str,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 100,
    fallback_overlap: int = 64,
    metadata: dict | None = None,
) -> list[DocumentChunk]:
    """
    Split text into chunks that respect document structure.

    Args:
        text: Text produced by pdf_parser or docx_parser (headings as
              markdown # markers).
        max_chunk_size: If a section body exceeds this, it is further
                        split with fixed-size chunking.
        min_chunk_size: Sections shorter than this are merged with the
                        following section to avoid tiny, context-free chunks.
        fallback_overlap: Overlap used when falling back to fixed-size.
        metadata: Base metadata attached to every chunk. heading,
                  heading_level, and chunk_index are added automatically.

    Returns:
        List of DocumentChunk objects. Use chunk.text_with_heading for
        embedding -- it prepends the section heading to the body text.

    Example:
        chunks = split_document_aware(
            text=docx_doc.text,
            max_chunk_size=1500,
            metadata={"doc_type": "policy", "client_id": None},
        )
        for c in chunks[:5]:
            print(f"  [{c.heading_level}] {c.heading!r:40s}  {c.char_length} chars")
    """
    sections = _extract_sections(text)
    if not sections:
        return []

    base_meta = dict(metadata or {})
    chunks: list[DocumentChunk] = []
    pending_text = ""
    pending_heading = ""
    pending_level = 0

    def flush(heading: str, body: str, level: int) -> None:
        """Turn accumulated text into one or more DocumentChunks."""
        if not body.strip():
            return
        if len(body) > max_chunk_size:
            # Fall back to fixed-size for long sections
            sub_chunks = split_fixed_size(
                body,
                chunk_size=max_chunk_size,
                chunk_overlap=fallback_overlap,
            )
            for sc in sub_chunks:
                chunk_meta = {
                    **base_meta,
                    "chunk_index": len(chunks),
                    "heading": heading,
                    "heading_level": level,
                }
                chunks.append(DocumentChunk(
                    text=sc.text,
                    chunk_index=len(chunks),
                    heading=heading,
                    heading_level=level,
                    metadata=chunk_meta,
                ))
        else:
            chunk_meta = {
                **base_meta,
                "chunk_index": len(chunks),
                "heading": heading,
                "heading_level": level,
            }
            chunks.append(DocumentChunk(
                text=body,
                chunk_index=len(chunks),
                heading=heading,
                heading_level=level,
                metadata=chunk_meta,
            ))

    for heading, body, level in sections:
        body = body.strip()

        # Merge short sections with the pending buffer
        if len(body) < min_chunk_size and body:
            pending_text = (pending_text + "\n\n" + body).strip() if pending_text else body
            if not pending_heading:
                pending_heading = heading
                pending_level = level
            continue

        # Flush pending buffer first
        if pending_text:
            flush(pending_heading, pending_text, pending_level)
            pending_text = ""
            pending_heading = ""
            pending_level = 0

        flush(heading, body, level)

    # Flush any remaining pending text
    if pending_text:
        flush(pending_heading, pending_text, pending_level)

    return chunks
