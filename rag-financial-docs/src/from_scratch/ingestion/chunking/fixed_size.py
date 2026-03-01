"""
Fixed-size chunking with configurable overlap.

The simplest chunking strategy: split text every N characters with an
overlap of K characters between consecutive chunks.

Why overlap?
  Without overlap, a sentence at a chunk boundary gets split in half.
  Overlap ensures boundary sentences appear fully in at least one chunk.

When to use:
  - Fast prototyping: no dependencies, deterministic
  - Homogeneous text (news, plain narrative reports)
  - When you need predictable uniform chunk sizes

Not ideal for:
  - Structured documents with headings (use document_aware instead)
  - Financial tables (use table_parser instead)

Rule of thumb: chunk_size=512 tokens ~ 2000 chars, overlap=10-15% of chunk_size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """A single chunk of text with positional metadata."""

    text: str
    chunk_index: int      # 0-based position in the document
    start_char: int       # character offset in the original text
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"TextChunk(index={self.chunk_index}, chars={self.char_length}, text={preview!r}...)"


def split_fixed_size(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[TextChunk]:
    """
    Split text into fixed-size character chunks with overlap.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Characters repeated at the start of the next chunk.
                       Must be less than chunk_size.
        metadata: Base metadata dict attached to every chunk.
                  chunk_index, start_char, end_char are added automatically.

    Returns:
        List of TextChunk objects in document order.

    Example:
        chunks = split_fixed_size(
            text=doc.full_text,
            chunk_size=512,
            chunk_overlap=64,
            metadata={"client_id": "acme_corp", "doc_type": "financial_report"},
        )
        avg = sum(c.char_length for c in chunks) / len(chunks)
        print(f"{len(chunks)} chunks, avg {avg:.0f} chars each")
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    if not text.strip():
        return []

    base_meta = dict(metadata or {})
    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_meta = {**base_meta, "chunk_index": len(chunks)}
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=start,
                end_char=end,
                metadata=chunk_meta,
            ))

        start += step

    return chunks


def split_fixed_size_by_words(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[TextChunk]:
    """
    Fixed-size chunking that never cuts in the middle of a word.

    Same logic as split_fixed_size, but chunk boundaries snap to the
    nearest whitespace. Prevents chunks ending mid-word like "reve-"
    or starting with "nue". Slightly less predictable sizes, cleaner text.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    if not text.strip():
        return []

    tokens: list[tuple[str, int]] = [
        (m.group(), m.start()) for m in re.finditer(r"\S+", text)
    ]
    if not tokens:
        return []

    base_meta = dict(metadata or {})
    chunks: list[TextChunk] = []
    i = 0

    while i < len(tokens):
        chunk_words: list[str] = []
        char_count = 0
        start_pos = tokens[i][1]
        j = i

        while j < len(tokens):
            word, _ = tokens[j]
            added = (1 if chunk_words else 0) + len(word)
            if chunk_words and char_count + added > chunk_size:
                break
            chunk_words.append(word)
            char_count += added
            j += 1

        chunk_text = " ".join(chunk_words).strip()
        end_pos = tokens[j - 1][1] + len(tokens[j - 1][0]) if j > i else start_pos

        if chunk_text:
            chunk_meta = {**base_meta, "chunk_index": len(chunks)}
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=start_pos,
                end_char=end_pos,
                metadata=chunk_meta,
            ))

        if j == i:
            i += 1
            continue

        words_consumed = j - i
        avg_word_len = char_count / words_consumed if words_consumed else 5
        overlap_words = max(1, int(chunk_overlap / max(avg_word_len, 1)))
        i += max(1, words_consumed - overlap_words)

    return chunks
