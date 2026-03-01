"""
Tests for all three chunking strategies (from scratch).

Tests are intentionally free of external dependencies:
- fixed_size: pure Python, no imports beyond stdlib
- document_aware: same
- semantic: mocked to avoid needing sentence-transformers in CI
"""

from __future__ import annotations

import pytest

SAMPLE_TEXT = """# Annual Report 2023

## Executive Summary

Acme Corp delivered strong results in fiscal year 2023. Revenue grew 18% year-over-year, reaching $42.5 million. This growth was driven by expansion in the enterprise segment and successful product launches in Q2 and Q3.

## Revenue Breakdown

Total revenue for 2023 was $42.5 million compared to $36.0 million in 2022. The enterprise segment accounted for 65% of total revenue. Gross margin improved to 72% from 68% in the prior year.

## Operating Expenses

Operating expenses increased by 12% to $28.3 million. The increase was primarily driven by headcount additions in engineering and sales. Research and development spending grew 22% to $8.1 million as we accelerated product investment.

## Outlook

We expect revenue growth of 20-25% in fiscal year 2024. Key drivers include new product releases scheduled for H1 2024 and continued expansion in the European market.
"""

SHORT_TEXT = "Hello world. This is a test."


class TestFixedSize:
    def test_returns_list_of_text_chunks(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size, TextChunk
        chunks = split_fixed_size(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)
        assert isinstance(chunks, list)
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_produces_multiple_chunks_for_long_text(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        chunks = split_fixed_size(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        chunk_size = 200
        chunks = split_fixed_size(SAMPLE_TEXT, chunk_size=chunk_size, chunk_overlap=20)
        for c in chunks:
            # Stripped chunks may be slightly under due to whitespace removal
            assert c.char_length <= chunk_size + 5  # small tolerance

    def test_overlap_creates_repeated_content(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        chunks = split_fixed_size(SAMPLE_TEXT, chunk_size=200, chunk_overlap=50)
        if len(chunks) >= 2:
            end_of_first = chunks[0].text[-30:]
            start_of_second = chunks[1].text[:50]
            # Some overlap should exist
            assert any(w in start_of_second for w in end_of_first.split() if len(w) > 3)

    def test_chunk_indices_are_sequential(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        chunks = split_fixed_size(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_metadata_attached_to_every_chunk(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        meta = {"client_id": "acme_corp", "doc_type": "financial_report"}
        chunks = split_fixed_size(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20, metadata=meta)
        for c in chunks:
            assert c.metadata["client_id"] == "acme_corp"
            assert c.metadata["doc_type"] == "financial_report"
            assert "chunk_index" in c.metadata

    def test_empty_text_returns_empty_list(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        assert split_fixed_size("") == []
        assert split_fixed_size("   \n  ") == []

    def test_overlap_gte_chunk_size_raises(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        with pytest.raises(ValueError):
            split_fixed_size(SAMPLE_TEXT, chunk_size=100, chunk_overlap=100)

    def test_short_text_produces_single_chunk(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size
        chunks = split_fixed_size(SHORT_TEXT, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert SHORT_TEXT.strip() in chunks[0].text

    def test_word_splitter_no_mid_word_cuts(self):
        from src.from_scratch.ingestion.chunking.fixed_size import split_fixed_size_by_words
        chunks = split_fixed_size_by_words(SAMPLE_TEXT, chunk_size=100, chunk_overlap=10)
        for c in chunks:
            # Each chunk should start and end at a word boundary
            assert not c.text[0].isspace()
            assert not c.text[-1].isspace()


class TestDocumentAware:
    def test_returns_document_chunks(self):
        from src.from_scratch.ingestion.chunking.document_aware import (
            split_document_aware, DocumentChunk,
        )
        chunks = split_document_aware(SAMPLE_TEXT)
        assert isinstance(chunks, list)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_headings_attached_to_chunks(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        chunks = split_document_aware(SAMPLE_TEXT)
        headings = [c.heading for c in chunks]
        # At least one chunk should have a heading
        assert any(h for h in headings)

    def test_heading_levels_correct(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        chunks = split_document_aware(SAMPLE_TEXT)
        for c in chunks:
            assert c.heading_level >= 0

    def test_text_with_heading_contains_heading(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        chunks = split_document_aware(SAMPLE_TEXT)
        for c in chunks:
            if c.heading:
                assert c.heading in c.text_with_heading

    def test_chunk_indices_sequential(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        chunks = split_document_aware(SAMPLE_TEXT)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_content_preserved(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        chunks = split_document_aware(SAMPLE_TEXT)
        all_text = " ".join(c.text for c in chunks)
        # Key phrases from the original should appear somewhere
        assert "42.5 million" in all_text
        assert "enterprise segment" in all_text

    def test_max_chunk_size_triggers_fallback(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        # With very small max_chunk_size, long sections get sub-split
        chunks = split_document_aware(SAMPLE_TEXT, max_chunk_size=100)
        for c in chunks:
            assert c.char_length <= 110  # small tolerance

    def test_metadata_attached(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        meta = {"doc_type": "policy"}
        chunks = split_document_aware(SAMPLE_TEXT, metadata=meta)
        for c in chunks:
            assert c.metadata["doc_type"] == "policy"

    def test_empty_text_returns_empty(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        assert split_document_aware("") == []

    def test_text_without_headings(self):
        from src.from_scratch.ingestion.chunking.document_aware import split_document_aware
        plain = "This is a paragraph.\n\nThis is another paragraph with more content here."
        chunks = split_document_aware(plain)
        assert len(chunks) >= 1
        # No heading found, heading should be empty string
        for c in chunks:
            assert c.heading_level == 0 or c.heading == ""


class TestSemanticChunker:
    """
    Semantic chunker tests use a mock to avoid needing sentence-transformers.
    The mock replaces the SentenceTransformer with a simple counter-based
    embedding that produces low similarity at predetermined positions.
    """

    def _mock_embeddings(self, sentences):
        """Return embeddings that create a split after sentence index 2."""
        import numpy as np
        embeddings = []
        for i, _ in enumerate(sentences):
            if i <= 1:
                embeddings.append(np.array([1.0, 0.0]))  # similar to each other
            elif i == 2:
                embeddings.append(np.array([0.0, 1.0]))  # different: triggers split
            else:
                embeddings.append(np.array([0.0, 1.0]))  # same as after-split
        return np.array(embeddings)

    def test_returns_semantic_chunks(self, monkeypatch):
        from src.from_scratch.ingestion.chunking import semantic as sem_module
        from src.from_scratch.ingestion.chunking.semantic import SemanticChunk

        class MockModel:
            def encode(self, sentences, show_progress_bar=False):
                return self._mock_embeddings(sentences)
            def _mock_embeddings(self, sentences):
                import numpy as np
                return self._outer._mock_embeddings(sentences)

        # Monkeypatch SentenceTransformer
        mock_self = self
        class MockST:
            def __init__(self, model_name):
                pass
            def encode(self, sentences, show_progress_bar=False):
                return mock_self._mock_embeddings(sentences)

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            MockST,
            raising=False,
        )

        chunks = sem_module.split_semantic(SAMPLE_TEXT, similarity_threshold=0.5)
        assert isinstance(chunks, list)
        assert all(isinstance(c, SemanticChunk) for c in chunks)

    def test_sentence_splitter_basic(self):
        from src.from_scratch.ingestion.chunking.semantic import _split_into_sentences
        sentences = _split_into_sentences("Hello world. How are you? Fine.")
        assert len(sentences) >= 1
        assert any("Hello" in s for s in sentences)

    def test_cosine_similarity(self):
        import numpy as np
        from src.from_scratch.ingestion.chunking.semantic import _cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6

        c = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, c)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        import numpy as np
        from src.from_scratch.ingestion.chunking.semantic import _cosine_similarity
        assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
