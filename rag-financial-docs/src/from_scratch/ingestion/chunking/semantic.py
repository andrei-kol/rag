"""
Semantic chunking: split by meaning, not by character count.

Instead of splitting at fixed character positions, we split where the
*meaning* of the text changes. The algorithm:

  1. Split text into sentences.
  2. Embed each sentence with a lightweight model (sentence-transformers).
  3. Compute cosine similarity between adjacent sentence embeddings.
  4. Wherever similarity drops below a threshold, insert a chunk boundary.
  5. Merge sentences within each boundary into one chunk.

Why this matters:
  Fixed-size chunking might split: "Revenue grew 18%. [SPLIT] This growth
  was driven by..." -- the second chunk loses its referent.
  Semantic chunking keeps topically coherent sentences together.

Trade-offs:
  - Requires a sentence-transformers model (first run downloads ~90 MB)
  - Slower than fixed-size (embedding each sentence has overhead)
  - Chunk sizes are variable -- can be very small or very large
  - Threshold tuning is required per domain

When to use:
  - Narrative text where topics shift gradually (policy docs, reports)
  - When retrieval quality matters more than indexing speed
  Not ideal for: very short documents, tables, code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemanticChunk:
    """A chunk whose boundaries are determined by semantic similarity drops."""

    text: str
    chunk_index: int
    sentences: list[str]          # individual sentences in this chunk
    avg_similarity: float         # mean similarity between adjacent sentences
    metadata: dict = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"SemanticChunk(index={self.chunk_index}, "
            f"sentences={len(self.sentences)}, "
            f"text={preview!r}...)"
        )


def _split_into_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter using punctuation heuristics.

    Good enough for English financial/legal text. For production,
    consider spaCy's sentencizer or NLTK's sent_tokenize.
    """
    # Split on . ! ? followed by whitespace and uppercase letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(])", text)
    # Also split on double newlines (paragraph breaks)
    sentences = []
    for fragment in raw:
        paragraphs = re.split(r"\n{2,}", fragment)
        sentences.extend(p.strip() for p in paragraphs if p.strip())
    return sentences


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def split_semantic(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.5,
    min_chunk_sentences: int = 1,
    max_chunk_sentences: int = 20,
    metadata: dict | None = None,
) -> list[SemanticChunk]:
    """
    Split text into semantically coherent chunks.

    A new chunk begins wherever the cosine similarity between adjacent
    sentence embeddings drops below `similarity_threshold`.

    Args:
        text: The text to split.
        model_name: sentence-transformers model to use for embeddings.
                    'all-MiniLM-L6-v2' is a good default: small and fast.
        similarity_threshold: Similarity below which we start a new chunk.
                              Range [0, 1]. Lower = fewer, larger chunks.
                              Typical range: 0.3 (coarse) to 0.7 (fine).
        min_chunk_sentences: Minimum sentences per chunk (prevents tiny chunks).
        max_chunk_sentences: Maximum sentences per chunk (prevents huge chunks).
        metadata: Base metadata attached to every chunk.

    Returns:
        List of SemanticChunk objects.

    Example:
        chunks = split_semantic(
            text=policy_text,
            similarity_threshold=0.5,
            metadata={"doc_type": "policy"},
        )
        print(f"{len(chunks)} semantic chunks")
        for c in chunks[:3]:
            print(f"  Sentences: {len(c.sentences)}, chars: {c.char_length}")
    """
    # Import here to avoid making sentence-transformers a hard dependency
    # for users who only use fixed-size or document-aware chunking.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for semantic chunking. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        base_meta = dict(metadata or {})
        base_meta["chunk_index"] = 0
        return [SemanticChunk(
            text=sentences[0],
            chunk_index=0,
            sentences=sentences,
            avg_similarity=1.0,
            metadata=base_meta,
        )]

    # Embed all sentences in one batch (much faster than one-by-one)
    model = SentenceTransformer(model_name)
    embeddings: np.ndarray = model.encode(sentences, show_progress_bar=False)

    # Compute similarity between each adjacent pair
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Build chunks: start a new chunk when similarity drops below threshold
    base_meta = dict(metadata or {})
    chunks: list[SemanticChunk] = []
    current_sentences: list[str] = [sentences[0]]
    current_sims: list[float] = []

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        should_split = (
            sim < similarity_threshold
            or len(current_sentences) >= max_chunk_sentences
        )
        too_small = len(current_sentences) < min_chunk_sentences

        if should_split and not too_small:
            # Finalise current chunk
            chunk_text = " ".join(current_sentences)
            avg_sim = float(np.mean(current_sims)) if current_sims else 1.0
            chunk_meta = {**base_meta, "chunk_index": len(chunks)}
            chunks.append(SemanticChunk(
                text=chunk_text,
                chunk_index=len(chunks),
                sentences=current_sentences[:],
                avg_similarity=avg_sim,
                metadata=chunk_meta,
            ))
            current_sentences = [next_sentence]
            current_sims = []
        else:
            current_sentences.append(next_sentence)
            current_sims.append(sim)

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        avg_sim = float(np.mean(current_sims)) if current_sims else 1.0
        chunk_meta = {**base_meta, "chunk_index": len(chunks)}
        chunks.append(SemanticChunk(
            text=chunk_text,
            chunk_index=len(chunks),
            sentences=current_sentences,
            avg_similarity=avg_sim,
            metadata=chunk_meta,
        ))

    return chunks
