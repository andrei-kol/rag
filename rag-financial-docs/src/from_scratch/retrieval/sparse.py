"""
Sparse Retrieval Module: BM25-based keyword search for financial documents.

BM25 is a probabilistic ranking function that scores documents based on how well they match
query terms. Unlike dense/semantic search which finds documents with similar meaning, BM25
excels at finding documents with exact keyword matches—perfect for ticker symbols, legal
terms, specific percentages, and article references.

Key differences from dense embeddings:
- Dense: Maps text to vectors; finds semantic similarity (e.g., "revenue grew" ≈ "sales increased")
- Sparse (BM25): Ranks based on term overlap; finds exact/exact-like matches (e.g., "AAPL", "18.3%")

BM25 Formula (simplified):
    Score(d, q) = Σ IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * (len(d) / avgLen)))
    
Where:
  - IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5)): penalizes common terms, boosts rare ones
  - TF(t,d) = frequency of term t in document d
  - k1: controls term saturation (1.5 = sweet spot; higher = more linear)
  - b: controls length normalization (0.75 = default; penalizes very long docs slightly)

When to use BM25:
  - Query contains ticker symbols: "AAPL", "MSFT", "GOOGL"
  - Query has specific numbers/percentages: "18.3%", "$15.2B", "Q3 2024"
  - Query has technical/legal terms: "force majeure", "EBITDA", "Section 4.2(b)"
  - Query is a code or reference: "10-K filing", "Clause 7(a)"

When to use Dense (embeddings):
  - Query is paraphrased or conceptual: "What's the company's financial health?"
  - Query uses synonyms: "revenue growth" vs "sales increase"
  - Query is multilingual: English query should find French document
"""

from dataclasses import dataclass
from typing import Optional
import string
import logging

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class SparseResult:
    """Result from a BM25 sparse retrieval query.
    
    Attributes:
        text: The matched document text (usually chunk.text or chunk.text_with_heading)
        score: BM25 relevance score (higher = better match)
        rank: Position in ranked results (1-indexed, so top result = rank 1)
        metadata: Optional dict with chunk metadata (e.g., source, page, chunk_id)
    """
    text: str
    score: float
    rank: int
    metadata: Optional[dict] = None


class BM25Retriever:
    """BM25-based sparse retriever for exact keyword matching in financial documents.
    
    This class implements the Okapi BM25 ranking function, which is particularly effective
    at finding documents with exact term matches (ticker symbols, percentages, legal terms).
    
    Attributes:
        k1: Term saturation parameter (default 1.5). Controls how much additional term
            occurrences matter. Higher values (e.g., 2.0) make the function more linear
            (each occurrence matters more); lower values (e.g., 0.5) saturate faster
            (first few occurrences matter most). 1.5 is the research-proven sweet spot.
            
        b: Length normalization parameter (default 0.75). Controls how much we penalize
            long documents. b=0 means no length normalization (long docs get no penalty);
            b=1 means full normalization (docs normalized to average length). 0.75 is a
            good compromise: long docs are penalized slightly but not harshly.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 retriever with tuning parameters.
        
        Args:
            k1: Term saturation parameter. Controls diminishing returns of additional
                term occurrences. Typical range: 1.2-2.0. Default 1.5 is research-optimal.
            b: Length normalization parameter. Range: 0.0-1.0. Higher values penalize
                longer documents more. Default 0.75 is standard in information retrieval.
        """
        self.k1 = k1
        self.b = b
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: Optional[list[list[str]]] = None
        self._chunks: Optional[list] = None
    
    @property
    def is_fitted(self) -> bool:
        """Check if the retriever has been fitted with a corpus.
        
        Returns:
            True if fit() has been called successfully, False otherwise.
        """
        return self._bm25 is not None
    
    def fit(self, chunks: list) -> None:
        """Build BM25 index from a list of document chunks.
        
        This method:
        1. Stores original chunks for later retrieval
        2. Tokenizes each chunk using _tokenize() (lowercasing, punctuation removal, stopword filtering)
        3. Builds BM25Okapi index from tokenized corpus
        4. Logs the process
        
        Args:
            chunks: List of chunk objects. Each chunk should have either:
                   - .text attribute (raw chunk text), or
                   - .text_with_heading attribute (preferred, includes heading context)
        
        Raises:
            ValueError: If chunks list is empty
            AttributeError: If chunks don't have required attributes
        """
        if not chunks:
            raise ValueError("Cannot fit BM25 retriever with empty chunk list")
        
        self._chunks = chunks
        
        # Choose which text field to use. text_with_heading includes section headers,
        # which provides valuable context for financial documents (e.g., "Income Statement" section)
        self._tokenized_corpus = []
        for chunk in chunks:
            text_to_tokenize = getattr(chunk, 'text_with_heading', None) or getattr(chunk, 'text', None)
            if text_to_tokenize is None:
                raise AttributeError(f"Chunk missing 'text' or 'text_with_heading' attribute")
            tokens = self._tokenize(text_to_tokenize)
            self._tokenized_corpus.append(tokens)
        
        # Initialize BM25Okapi with our tokenized corpus and tuning parameters
        # BM25Okapi is the "Okapi Best Match 25" variant, the most widely-used BM25 implementation
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Fitted BM25 retriever with {len(chunks)} chunks (k1={self.k1}, b={self.b})")
    
    def search(self, query: str, top_k: int = 5) -> list[SparseResult]:
        """Search the corpus for chunks matching the query using BM25 ranking.
        
        This method:
        1. Tokenizes the query the same way as corpus documents (for consistency)
        2. Computes BM25 scores for all documents
        3. Filters out zero-score results (documents with no term overlap)
        4. Ranks results and returns top_k as SparseResult objects
        
        Args:
            query: Search query string (e.g., "AAPL earnings 18.3%")
            top_k: Number of top results to return. Default 5.
        
        Returns:
            List of SparseResult objects, ranked by BM25 score (highest first).
            Empty list if no matches found.
        
        Raises:
            RuntimeError: If retriever hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever not fitted. Call fit() first.")
        
        # Tokenize query using the same preprocessing as the corpus
        query_tokens = self._tokenize(query)
        
        # Compute BM25 scores for all documents in corpus
        # get_scores() returns a numpy array of scores, one per document
        scores = self._bm25.get_scores(query_tokens)
        
        # Create list of (index, score) tuples, filtering out zero scores
        # Zero scores mean the document shares no terms with the query—not useful
        scored_docs = [(i, float(score)) for i, score in enumerate(scores) if score > 0]
        
        # Sort by score descending (best matches first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to SparseResult objects, keeping only top_k
        results = []
        for rank, (chunk_idx, score) in enumerate(scored_docs[:top_k], start=1):
            chunk = self._chunks[chunk_idx]
            # Prefer text_with_heading for context, fall back to text
            result_text = getattr(chunk, 'text_with_heading', None) or getattr(chunk, 'text', '')
            
            # Extract metadata if available (source, page number, chunk ID, etc.)
            metadata = getattr(chunk, 'metadata', {}) or {}
            
            results.append(SparseResult(
                text=result_text,
                score=score,
                rank=rank,
                metadata=metadata
            ))
        
        logger.debug(f"BM25 search for '{query}' returned {len(results)} results")
        return results
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing and querying.
        
        This simple tokenization:
        1. Converts to lowercase (case-insensitive matching)
        2. Removes punctuation (to match financial numbers like "$15.2B" with "15.2B")
        3. Splits on whitespace
        4. Filters out English stopwords (very common words that add noise, not signal)
        
        Note: This is a simple approach suitable for English financial documents.
        For production systems, consider NLTK's better stopword list or spaCy.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of lowercased, punctuation-free tokens without stopwords
        """
        # Hardcoded English stopwords: common words that appear in nearly every document
        # These add noise to BM25 scoring (every doc matches "the"), so we filter them
        # For a production system, consider using NLTK's stopwords: nltk.corpus.stopwords.words('english')
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'as',
            'it', 'its', 'that', 'this', 'which', 'who', 'when', 'where', 'why', 'how'
        }
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Remove punctuation: replace all punctuation chars with space
        # This allows "$15.2B" to match with "15.2B", "15 2 B", or "152B"
        # while preserving important content words
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
        # Split on whitespace to get tokens
        tokens = text.split()
        
        # Filter: keep only tokens that are NOT stopwords and are non-empty
        filtered_tokens = [t for t in tokens if t and t not in stopwords]
        
        return filtered_tokens


def build_bm25_retriever(chunks: list, k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """Convenience function to create and fit a BM25 retriever in one call.
    
    This is a helper for the common pattern of:
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.fit(chunks)
    
    Args:
        chunks: List of document chunks to index
        k1: Term saturation parameter (see BM25Retriever.__init__)
        b: Length normalization parameter (see BM25Retriever.__init__)
    
    Returns:
        Fitted BM25Retriever instance ready for searching
    
    Example:
        >>> retriever = build_bm25_retriever(chunks, k1=1.5, b=0.75)
        >>> results = retriever.search("AAPL earnings per share", top_k=5)
    """
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.fit(chunks)
    return retriever


# ============================================================================
# WHEN TO USE SPARSE (BM25) vs DENSE (embeddings):
# ============================================================================
#
# BM25 (Sparse) wins when query contains:
#   - Exact ticker symbols: "AAPL", "MSFT", "GOOGL", "BRK.B"
#   - Specific numbers/percentages: "18.3%", "$15.2B", "Q3 2024", "2.5x"
#   - Legal/technical terms: "force majeure", "EBITDA", "covenant", "subordinated"
#   - Article/section references: "Section 4.2", "Clause 7(b)", "10-K filing", "Form 8-K"
#   - Proper nouns: "Federal Reserve", "Treasury Department", "Moody's", "S&P"
#
# Dense (Embeddings) wins when query is:
#   - Paraphrased/loose: "revenue went up" should match "sales increased"
#   - Conceptual/semantic: "financial health" matches docs about "liquidity and solvency"
#   - Multilingual: English query should find French or Chinese document with same meaning
#   - Complex reasoning: "companies that are struggling" matches docs with signs of distress
#
# HYBRID APPROACH (best of both worlds):
#   - Use both BM25 and dense embeddings
#   - Combine scores (simple: average, weighted average, or max)
#   - Example: "AAPL revenue growth" → BM25 finds AAPL docs, dense finds "growth" concepts
#   - See hybrid.py for implementation
#
# Example scenario:
#   Query: "AAPL earnings report Q3 2024 shows 18.3% growth"
#   
#   BM25 strength: Instantly finds docs mentioning AAPL, Q3, 2024, 18.3, growth
#   BM25 weakness: Doesn't understand that "earnings report" ≈ "10-Q filing"
#   
#   Dense strength: Understands "earnings report" ≈ "10-Q", "growth" ≈ "increase"
#   Dense weakness: Might match generic "growth" docs unrelated to AAPL
#   
#   Hybrid result: BM25 ensures AAPL is present, dense ensures semantic relevance
#
# ============================================================================
