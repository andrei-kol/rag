"""
Reranking module for precise document scoring in RAG pipelines.

WHAT IS RERANKING?
Reranking is a second-pass scoring mechanism using a more powerful model to refine
initial retrieval results. Instead of scoring all documents, we use a cheaper model
(like BM25 or dense embeddings) to retrieve a larger set, then apply a stronger model
to narrow down to the most relevant results.

PIPELINE FLOW:
    1. Dense or sparse retrieval → get top 20 candidates cheaply
    2. Cross-encoder reranking → score all 20, keep top 3 precisely
    3. Return top-3 for LLM context

WHY CROSS-ENCODERS FOR RERANKING?
- Bi-encoders (dense embeddings): query and document encoded SEPARATELY
  → Misses interaction patterns (word alignments, semantic mismatches)
  → Fast but less accurate
  
- Cross-encoders: query and document scored JOINTLY
  → Sees query-document pairs as a unit
  → Captures fine-grained semantic alignment
  → Slower (~10-100x) but much more accurate (2-5% improvements common)

TRADE-OFF:
- Cost: ~100ms per query (for top-20 candidates) vs ~1ms for dense lookup
- Benefit: Significantly better ranking precision, especially for edge cases
- Solution: Only rerank the top-20, not all documents in index

MODEL:
- cross-encoder/ms-marco-MiniLM-L-6-v2: 80MB, optimized for relevance ranking
- Trained on MS MARCO dataset (900K+ query-document pairs)
- Fast enough for real-time use (~10-50 docs per second on CPU)

WHY RERANKING MATTERS - EXAMPLE:
    Query: "What are the financial risks of holding Treasury bonds?"
    
    Initial ranking (dense embeddings):
    1. [bond portfolio management strategies]  score=0.82
    2. [interest rate risk for bond investors]  score=0.81
    3. [Treasury bond yield curve analysis]     score=0.79
    4. [credit default swaps for risk mitigation] score=0.78
    5. [inflation risk in long-term securities] score=0.75
    
    After reranking (cross-encoder):
    1. [interest rate risk for bond investors]  score=0.94  (was #2)
    2. [inflation risk in long-term securities] score=0.91  (was #5)
    3. [credit default swaps for risk mitigation] score=0.88  (was #4)
    
    The reranker correctly identified that documents about RISKS are more relevant
    than general bond management. Dense embeddings missed the semantic focus.
"""

from dataclasses import dataclass
from typing import Optional, Union, List, Any, Dict


@dataclass
class RankedChunk:
    """
    A document chunk with reranking scores and rank changes.
    
    Attributes:
        text: The actual document text.
        cross_encoder_score: Relevance score from cross-encoder (0-1, higher is better).
        original_rank: Position in the initial retrieval list (1-indexed).
        new_rank: Position after reranking (1-indexed).
        metadata: Dictionary with original chunk metadata (source, page, etc).
    """
    text: str
    cross_encoder_score: float
    original_rank: int
    new_rank: int
    metadata: Dict[str, Any]


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    
    A cross-encoder jointly encodes query-document pairs, producing a relevance score
    that better captures semantic alignment than independent embeddings. This class
    handles lazy model loading and provides efficient reranking.
    
    Attributes:
        model_name: HuggingFace model identifier for the cross-encoder.
        top_n: Default number of results to return.
        _model: Lazy-loaded cross-encoder model (None until first use).
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3
    ):
        """
        Initialize the reranker with lazy model loading.
        
        Args:
            model_name: HuggingFace model identifier. Default is optimized for
                        relevance ranking on MS MARCO dataset.
            top_n: Default number of reranked results to return.
        """
        self.model_name = model_name
        self.top_n = top_n
        self._model = None
    
    def _load_model(self):
        """Lazily load the cross-encoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
    
    def _get_text(self, chunk: Union[str, Any]) -> str:
        """
        Extract text from a chunk, handling multiple formats.
        
        Tries attributes in order of preference:
        1. .text_with_heading (chunk with heading prepended)
        2. .text (standard chunk attribute)
        3. Direct string conversion
        
        Args:
            chunk: A string or object with text content.
        
        Returns:
            The extracted text content.
        """
        if isinstance(chunk, str):
            return chunk
        if hasattr(chunk, 'text_with_heading'):
            return chunk.text_with_heading
        if hasattr(chunk, 'text'):
            return chunk.text
        return str(chunk)
    
    def rerank(
        self,
        query: str,
        chunks: List[Union[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RankedChunk]:
        """
        Rerank chunks by cross-encoder relevance scores.
        
        Process:
        1. Extract text from each chunk (handles multiple formats)
        2. Create (query, document_text) pairs
        3. Score all pairs with cross-encoder
        4. Sort by score and return top_n with rank metadata
        
        Args:
            query: The user query or search text.
            chunks: List of documents/chunks to rerank. Can be strings or objects
                    with .text or .text_with_heading attributes.
            top_n: Number of results to return. If None, uses self.top_n.
        
        Returns:
            List of RankedChunk objects sorted by cross_encoder_score (descending),
            with original and new rank information.
        
        Example:
            >>> reranker = CrossEncoderReranker()
            >>> chunks = ["doc1", "doc2", "doc3"]
            >>> results = reranker.rerank("financial risk", chunks, top_n=2)
            >>> for r in results:
            ...     print(f"{r.new_rank}. {r.text[:30]}... (score={r.cross_encoder_score:.2f})")
        """
        self._load_model()
        top_n = top_n or self.top_n
        
        # Extract text and build query-document pairs
        texts = [self._get_text(chunk) for chunk in chunks]
        pairs = [[query, text] for text in texts]
        
        # Score with cross-encoder
        scores = self._model.predict(pairs)
        
        # Build RankedChunk objects with metadata
        ranked_chunks = []
        for i, (chunk, text, score) in enumerate(zip(chunks, texts, scores)):
            # Extract metadata if the chunk is an object
            metadata = {}
            if hasattr(chunk, 'metadata'):
                metadata = chunk.metadata
            elif hasattr(chunk, 'source'):
                metadata = {'source': chunk.source}
            
            ranked_chunks.append(
                RankedChunk(
                    text=text,
                    cross_encoder_score=float(score),
                    original_rank=i + 1,
                    new_rank=0  # Will be set after sorting
                )
            )
        
        # Sort by cross-encoder score (descending)
        ranked_chunks.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        
        # Assign new ranks and truncate to top_n
        for i, chunk in enumerate(ranked_chunks[:top_n]):
            chunk.new_rank = i + 1
        
        return ranked_chunks[:top_n]


def rerank_chunks(
    query: str,
    chunks: List[Union[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 3
) -> List[RankedChunk]:
    """
    Standalone function to rerank chunks without managing a reranker instance.
    
    Convenience function that creates a reranker, reranks, and returns results.
    For repeated use, prefer creating a CrossEncoderReranker instance to reuse
    the loaded model across multiple queries.
    
    Args:
        query: The search query or context.
        chunks: List of documents to rerank.
        model_name: Cross-encoder model identifier.
        top_n: Number of results to return.
    
    Returns:
        List of RankedChunk objects sorted by relevance.
    
    Example:
        >>> results = rerank_chunks(
        ...     "bond interest rate risk",
        ...     chunk_list,
        ...     top_n=3
        ... )
    """
    reranker = CrossEncoderReranker(model_name=model_name, top_n=top_n)
    return reranker.rerank(query=query, chunks=chunks, top_n=top_n)
