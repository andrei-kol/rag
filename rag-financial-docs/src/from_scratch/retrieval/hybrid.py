"""
Hybrid search combining dense and sparse retrieval using Reciprocal Rank Fusion (RRF).

WHAT IS HYBRID SEARCH?
    Hybrid search combines two complementary retrieval approaches:
    - Dense (semantic): Uses embeddings to find semantically similar documents
    - Sparse (keyword): Uses BM25 or keyword matching to find exact term matches
    
    Example: Query "financial risk assessment"
        Dense alone: finds "risk management frameworks" (similar concept)
        Sparse alone: finds "risk" or "assessment" keywords
        Hybrid: combines semantic relevance + keyword precision

WHY RECIPROCAL RANK FUSION (RRF)?
    Instead of combining scores (hard to normalize across different retrievers),
    RRF combines RANKS. This is robust because:
    
    - Dense scores (cosine similarity) range [0, 1]
    - Sparse scores (BM25) range [0, ~25]
    - Different scales make naive averaging meaningless
    
    RRF formula: score(d) = sum(1 / (k + rank(d)))
    Where k is a constant (default 60), and rank is 1-indexed position
    
    Example with k=60:
        rank 1: 1/(60+1) = 0.0164
        rank 2: 1/(60+2) = 0.0161
        rank 10: 1/(60+10) = 0.0143
    
    Small but consistent differences reward docs consistently ranked high.

WHY k=60?
    - Empirically found to give robust fusion across diverse datasets
    - Smooths differences between ranks (rank 1 vs rank 2 < rank 10 vs rank 11)
    - Prevents single retriever from dominating the result
    - Research: "Reciprocal Rank Fusion outperforms other combination methods"

WHEN DOES HYBRID WIN?
    - Multi-concept queries: "inflation AND interest rates AND employment"
    - Named entities + concepts: "Tesla quarterly earnings analysis"
    - Time-sensitive + semantic: "recent market volatility assessment"
    - Rare + important keywords: "ESG compliance" (keyword precision)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dense import DenseRetriever
    from .sparse import BM25Retriever


@dataclass
class HybridResult:
    """Result from hybrid retrieval combining dense and sparse search."""
    
    text: str
    """The document chunk or passage text."""
    
    dense_rank: int | None = None
    """Rank from dense retriever (1-indexed), None if not retrieved."""
    
    sparse_rank: int | None = None
    """Rank from sparse retriever (1-indexed), None if not retrieved."""
    
    rrf_score: float = 0.0
    """RRF combined score: sum of weighted reciprocal ranks."""
    
    metadata: dict = field(default_factory=dict)
    """Document metadata (source, doc_type, year, etc.)."""
    
    def __lt__(self, other: "HybridResult") -> bool:
        """Sort by rrf_score descending (higher is better)."""
        return self.rrf_score > other.rrf_score


class HybridRetriever:
    """
    Combine dense (semantic) and sparse (keyword) retrieval using RRF.
    
    Usage:
        hybrid = HybridRetriever(
            dense_retriever=embedding_based_retriever,
            sparse_retriever=bm25_retriever,
            rrf_k=60,
            dense_weight=0.7,
            sparse_weight=0.3
        )
        results = hybrid.search("financial risk assessment", top_k=5)
    """
    
    def __init__(
        self,
        dense_retriever: "DenseRetriever",
        sparse_retriever: "BM25Retriever",
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense (embedding-based) retriever
            sparse_retriever: Sparse (BM25) retriever
            rrf_k: Constant in RRF formula (default 60, empirically optimal)
            dense_weight: Weight for dense retriever scores (default 0.7)
            sparse_weight: Weight for sparse retriever scores (default 0.3)
            
        Raises:
            ValueError: If weights don't sum to 1.0 (approximately)
        """
        if not (0.99 <= dense_weight + sparse_weight <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {dense_weight + sparse_weight}"
            )
        
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        """
        Search using RRF to combine dense and sparse results.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of HybridResult sorted by rrf_score (descending)
        """
        # Fetch more candidates from each retriever
        # (some may not appear in both, so we oversample)
        expanded_k = max(top_k * 2, 20)
        
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, top_k=expanded_k)
        sparse_results = self.sparse_retriever.search(query, top_k=expanded_k)
        
        # Create a mapping of text -> result info
        # Store (dense_rank, sparse_rank, text, metadata)
        result_map: dict[str, tuple[int | None, int | None, str, dict]] = {}
        
        # Add dense results
        for rank, result in enumerate(dense_results, start=1):
            text = result.text
            metadata = getattr(result, "metadata", {})
            result_map[text] = (rank, None, text, metadata)
        
        # Add sparse results, merging with existing dense results
        for rank, result in enumerate(sparse_results, start=1):
            text = result.text
            metadata = getattr(result, "metadata", {})
            
            if text in result_map:
                # Already have dense rank, add sparse rank
                dense_rank, _, _, existing_metadata = result_map[text]
                # Merge metadata, preferring existing
                merged_metadata = {**metadata, **existing_metadata}
                result_map[text] = (dense_rank, rank, text, merged_metadata)
            else:
                # Only in sparse results
                result_map[text] = (None, rank, text, metadata)
        
        # Calculate RRF scores
        hybrid_results: list[HybridResult] = []
        
        for text, (dense_rank, sparse_rank, _, metadata) in result_map.items():
            # Calculate weighted RRF score
            score = 0.0
            
            if dense_rank is not None:
                dense_component = self.dense_weight * (
                    1.0 / (self.rrf_k + dense_rank)
                )
                score += dense_component
            
            if sparse_rank is not None:
                sparse_component = self.sparse_weight * (
                    1.0 / (self.rrf_k + sparse_rank)
                )
                score += sparse_component
            
            hybrid_results.append(
                HybridResult(
                    text=text,
                    dense_rank=dense_rank,
                    sparse_rank=sparse_rank,
                    rrf_score=score,
                    metadata=metadata,
                )
            )
        
        # Sort by RRF score (descending) and return top_k
        hybrid_results.sort(reverse=True, key=lambda r: r.rrf_score)
        return hybrid_results[:top_k]


def hybrid_search(
    query: str,
    dense_retriever: "DenseRetriever",
    sparse_retriever: "BM25Retriever",
    top_k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[HybridResult]:
    """
    Convenience function for one-off hybrid searches.
    
    Use HybridRetriever class for repeated searches (more efficient).
    
    Args:
        query: Search query string
        dense_retriever: Dense (embedding-based) retriever
        sparse_retriever: Sparse (BM25) retriever
        top_k: Number of results to return (default 5)
        rrf_k: RRF constant (default 60)
        dense_weight: Weight for dense results (default 0.7)
        sparse_weight: Weight for sparse results (default 0.3)
        
    Returns:
        List of HybridResult sorted by rrf_score (descending)
        
    Example:
        results = hybrid_search(
            "financial risk assessment",
            dense_ret,
            bm25_ret,
            top_k=10
        )
    """
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    return retriever.search(query, top_k=top_k)


# WHY RRF INSTEAD OF SCORE NORMALIZATION?
#
# Option A (naive score normalization): min-max scale to [0,1] then average
#   Problem:
#     - BM25 scores range 0-25, cosine similarity 0-1
#     - Different scales make averaging meaningless
#     - Query "the" vs "blockchain" have vastly different BM25 distributions
#     - Requires dataset-specific calibration
#
# Option B (RRF - use ranks, not scores):
#   Advantages:
#     - 1/(60 + rank) gives: rank1=0.0164, rank2=0.0161, rank10=0.0143
#     - Small but consistent differences reward consistently high-ranked docs
#     - No calibration needed across different retrievers or datasets
#     - Robust to outlier scores
#     - Proven effective in information retrieval literature
#
# Real example:
#   Dense score for doc A: 0.85 (4th place in 100 docs)
#   BM25 score for doc A: 2.3 (3rd place in 100 docs)
#   
#   Naive approach: (0.85 + 2.3)/2 = 1.575 (meaningless without context)
#   RRF approach: 1/64 + 1/63 = 0.0156 + 0.0159 = 0.0315 (comparable to others)
