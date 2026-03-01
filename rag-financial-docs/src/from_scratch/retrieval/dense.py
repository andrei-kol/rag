"""
Dense vector retrieval using Qdrant and OpenAI embeddings.

This module provides a production-grade dense retrieval system that leverages
vector similarity search for semantic document retrieval. It uses OpenAI's
text-embedding-3-small model for generating high-quality embeddings and Qdrant
as a scalable vector database.

Key design decisions:
- OpenAI embeddings: State-of-the-art semantic understanding with 1536 dimensions
- Qdrant: Efficient vector search with HNSW indexing, supporting metadata filtering
- Lazy initialization: Defers client creation until first use (memory efficient)
- Batch processing: Handles large document sets with configurable batch sizes and progress tracking
- Type safety: Dataclasses for type hints and validation
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from openai import OpenAI
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """
    Represents a text chunk with its embedding vector.
    
    This dataclass encapsulates a chunk of text along with its corresponding
    dense vector representation, making it suitable for storage in vector
    databases like Qdrant.
    
    Attributes:
        text: The original text content of the chunk. Primary content for display
              and context.
        chunk_index: Zero-based index of this chunk in the original document sequence.
                     Used for tracking chunk order and reconstructing document structure.
        embedding: List of floats representing the dense vector embedding.
                   For OpenAI text-embedding-3-small, this is a 1536-dimensional vector.
                   These embeddings capture semantic meaning in high-dimensional space,
                   enabling similarity-based search.
        metadata: Dictionary of arbitrary metadata associated with the chunk.
                 Typical keys: 'document_id', 'page_number', 'source_file', 'timestamp'.
                 Metadata enables filtered search and result enrichment.
    """
    text: str
    chunk_index: int
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """
    Represents a chunk retrieved from the vector database.
    
    This dataclass wraps search results with ranking information, making it
    easy to understand relevance scores and result ordering.
    
    Attributes:
        text: The original text content of the retrieved chunk.
        score: Similarity score between 0 and 1 (for cosine distance on normalized vectors).
               Higher scores indicate greater semantic similarity to the query.
               For cosine distance: score = 1 - distance, where distance in [0, 2].
        metadata: Dictionary containing chunk metadata from indexing time.
                 Preserves all indexing-time metadata for result context.
        rank: Position in the result set (1-indexed for user clarity).
              rank=1 indicates the most relevant result.
    """
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 1


class DenseRetriever:
    """
    Production-grade dense vector retriever using Qdrant and OpenAI embeddings.
    
    This class manages the complete lifecycle of dense retrieval: embedding generation,
    collection management, and semantic search. It uses lazy initialization to defer
    client creation until first use.
    
    Design rationale:
    - Lazy initialization: Clients are created only when needed, avoiding unnecessary
      connections and API initialization. This enables efficient testing and lightweight
      instantiation.
    - OpenAI text-embedding-3-small: Provides 1536-dimensional embeddings with excellent
      cost-performance ratio and strong semantic understanding.
    - Cosine distance in Qdrant: Ideal for normalized embeddings, measuring angle between
      vectors. More robust to vector magnitude than L2 distance.
    - Metadata filtering: Enables efficient pre-filtering before similarity computation,
      reducing search space and supporting structured queries.
    
    Attributes:
        collection_name: Primary collection for operations (can be overridden per method).
        qdrant_url: Connection URL for Qdrant instance (e.g., 'http://localhost:6333').
        openai_api_key: API key for OpenAI embeddings API.
        model_name: OpenAI embedding model identifier. text-embedding-3-small provides
                   1536 dimensions and excellent performance.
        top_k: Default number of results to retrieve per search query.
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_url: str,
        openai_api_key: str,
        model_name: str = "text-embedding-3-small",
        top_k: int = 5,
    ):
        """
        Initialize DenseRetriever with configuration.
        
        Args:
            collection_name: Name of the primary Qdrant collection for this retriever.
            qdrant_url: URL to Qdrant instance (e.g., 'http://localhost:6333').
            openai_api_key: OpenAI API key for embeddings.
            model_name: OpenAI embedding model. Defaults to 'text-embedding-3-small'
                       which produces 1536-dimensional vectors.
            top_k: Default number of top results to return from searches. Can be
                  overridden per search call.
        
        Note:
            QdrantClient and OpenAI clients are lazily initialized on first use
            to avoid unnecessary connections during instantiation.
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.top_k = top_k
        
        # Lazy initialization: clients created on first use
        self._qdrant_client: Optional[QdrantClient] = None
        self._openai_client: Optional[OpenAI] = None
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """
        Lazily initialize and return Qdrant client.
        
        Returns:
            QdrantClient instance connected to configured qdrant_url.
        
        Design note:
            Lazy initialization ensures we only connect to Qdrant when actually needed.
            This improves startup time and enables graceful handling of unavailable services.
        """
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(url=self.qdrant_url)
            logger.debug(f"Initialized Qdrant client for {self.qdrant_url}")
        return self._qdrant_client
    
    @property
    def openai_client(self) -> OpenAI:
        """
        Lazily initialize and return OpenAI client.
        
        Returns:
            OpenAI client instance configured with provided API key.
        
        Design note:
            Lazy initialization defers API client creation and key validation until needed,
            reducing startup overhead and improving testability.
        """
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.openai_api_key)
            logger.debug("Initialized OpenAI client")
        return self._openai_client
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed.
        
        Returns:
            List of floats representing the embedding (1536 dimensions for
            text-embedding-3-small).
        
        Raises:
            Exception: If OpenAI API call fails.
        
        Design note:
            Single embeddings use the straightforward OpenAI API. For batch operations,
            use embed_batch() which may offer rate limit advantages.
        """
        response = self.openai_client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching and progress tracking.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to embed per API call. Default 100 balances
                       API efficiency and token limit considerations (text-embedding-3-small
                       supports batch sizes up to ~2000 texts, but 100 is conservative).
        
        Returns:
            List of embeddings in the same order as input texts. Each embedding is
            a list of 1536 floats for text-embedding-3-small.
        
        Design rationale:
            Batch processing is more efficient than individual requests:
            - Reduces number of API calls (1 call per ~batch_size texts vs 1 per text)
            - Improves throughput and reduces overall latency
            - Batch_size=100 is conservative; OpenAI supports larger batches but this
              avoids potential token limit issues while maintaining good efficiency
        
        Implementation note:
            Progress bar shows batch processing progress. Embeddings are collected
            in order to maintain input-output correspondence.
        """
        embeddings = []
        
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Embedding texts",
            unit="batch",
        ):
            batch_texts = texts[i : i + batch_size]
            response = self.openai_client.embeddings.create(
                model=self.model_name,
                input=batch_texts,
            )
            # OpenAI returns embeddings in input order, so we can append directly
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def index_chunks(
        self,
        chunks: list,
        collection_name: str | None = None,
    ) -> int:
        """
        Index a list of chunks into the Qdrant collection.
        
        This method handles the complete indexing pipeline: extracting text from chunk
        objects, generating embeddings, creating collections if needed, and upserting
        points to Qdrant.
        
        Args:
            chunks: List of chunk objects. Each chunk must have:
                   - .text or .text_with_heading attribute (tried in that order)
                   - .metadata attribute (dict of arbitrary metadata)
            collection_name: Qdrant collection to index into. If None, uses the
                           retriever's primary collection_name.
        
        Returns:
            Count of chunks successfully indexed.
        
        Raises:
            AttributeError: If chunks lack required .text/.text_with_heading or .metadata.
            Exception: If Qdrant operations fail.
        
        Design decisions:
            
            1. Text extraction (.text_with_heading fallback):
               Some chunking strategies include document structure in a 'text_with_heading'
               field that includes section headers for context. We prefer this when
               available, falling back to plain .text for compatibility.
               
            2. Vector dimension = 1536:
               OpenAI's text-embedding-3-small produces 1536-dimensional vectors.
               This is a fixed architectural constraint of the model.
               
            3. Cosine distance:
               Cosine similarity is ideal for normalized embeddings (which OpenAI produces).
               It measures angle between vectors, making it robust to magnitude variations.
               For normalized vectors, cosine distance = 1 - similarity, ranging [0, 2].
               
            4. Collection creation with VectorParams:
               If collection doesn't exist, we create it with:
               - vector_size=1536 (matching text-embedding-3-small output)
               - distance=Distance.COSINE (optimal for normalized embeddings)
               - This enables efficient HNSW indexing by Qdrant.
               
            5. Batch upserting via PointStruct:
               Qdrant's upsert operation accepts multiple points at once, more efficient
               than individual inserts. PointStruct encapsulates:
               - id: Auto-incremented, globally unique point identifier
               - vector: The 1536-dimensional embedding
               - payload: Dictionary of chunk text + metadata for retrieval
               
            6. Point IDs are auto-generated starting from 1 for each collection.
               This is safe because IDs are scoped to collections.
        
        Implementation note:
            Embeddings are generated in batches for efficiency. Progress is shown via tqdm.
        """
        if collection_name is None:
            collection_name = self.collection_name
        
        # Extract text and metadata from chunks
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # Try text_with_heading first (includes structure/headers), fall back to text
            if hasattr(chunk, "text_with_heading"):
                text = chunk.text_with_heading
            else:
                text = chunk.text
            
            texts.append(text)
            metadatas.append(chunk.metadata)
        
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embed_batch(texts)
        
        # Check if collection exists, create if not
        try:
            self.qdrant_client.get_collection(collection_name)
            logger.debug(f"Collection {collection_name} exists")
        except Exception:
            # Collection doesn't exist, create it
            logger.info(
                f"Creating collection {collection_name} with vector_size=1536 "
                f"(text-embedding-3-small output dimension)"
            )
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small produces 1536-dim vectors
                    distance=Distance.COSINE,  # Cosine ideal for normalized embeddings
                ),
            )
        
        # Create PointStruct objects for upserting
        points = []
        for idx, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Payload includes both text (for display) and metadata (for filtering)
            payload = {"text": text, **metadata}
            
            point = PointStruct(
                id=idx + 1,  # 1-indexed point IDs
                vector=embedding,
                payload=payload,
            )
            points.append(point)
        
        # Upsert all points at once
        logger.info(f"Upserting {len(points)} points to {collection_name}")
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )
        
        logger.info(f"Successfully indexed {len(chunks)} chunks")
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search for chunks semantically similar to the query.
        
        Args:
            query: Search query string. Will be embedded and compared against indexed chunks.
            top_k: Number of results to return. If None, uses retriever's default top_k.
            filter_metadata: Dictionary for metadata filtering. Keys match payload metadata keys,
                           values are matched exactly. Example: {'source_file': 'earnings.pdf'}
                           Only chunks matching ALL filter conditions are considered.
        
        Returns:
            List of RetrievedChunk objects sorted by relevance (highest score first),
            with rank assigned in result order (1-indexed).
        
        Raises:
            Exception: If Qdrant search fails.
        
        Design note on metadata filtering:
            Qdrant supports efficient metadata filtering via the Filter API. This enables
            pre-filtering the search space, reducing computation and improving latency.
            Filters are applied server-side, before similarity computation, making them
            highly efficient for large collections.
        
        Implementation note:
            1. Query is embedded using the same model as indexed texts
            2. Qdrant performs similarity search using cosine distance (1 - similarity)
            3. Results are returned with distances; we convert to scores (1 - distance) for
               intuitive interpretation where higher = more relevant
            4. Metadata filters are converted to Qdrant Filter objects if provided
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.debug(f"Embedding query: {query[:100]}...")
        query_embedding = self.embed_text(query)
        
        # Build metadata filter if provided
        filter_obj = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                # MatchValue handles comparison; FieldCondition wraps it
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                conditions.append(condition)
            
            # If multiple filters, use AND logic (all must match)
            if len(conditions) == 1:
                filter_obj = Filter(must=conditions)
            else:
                filter_obj = Filter(must=conditions)
            
            logger.debug(f"Applying metadata filter: {filter_metadata}")
        
        # Perform search
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
        )
        
        # Convert Qdrant results to RetrievedChunk objects
        retrieved_chunks = []
        for rank, result in enumerate(results, start=1):
            # Score = 1 - distance for cosine (converts distance to similarity)
            # Qdrant returns scores for similarity search; use directly
            score = result.score
            
            # Extract text and metadata from payload
            text = result.payload.pop("text")
            metadata = result.payload  # Remaining payload is metadata
            
            chunk = RetrievedChunk(
                text=text,
                score=score,
                metadata=metadata,
                rank=rank,
            )
            retrieved_chunks.append(chunk)
        
        logger.debug(
            f"Search returned {len(retrieved_chunks)} results for query: {query[:100]}..."
        )
        return retrieved_chunks
    
    def delete_collection(self, collection_name: str | None = None) -> bool:
        """
        Delete a Qdrant collection.
        
        Args:
            collection_name: Collection to delete. If None, uses retriever's primary collection.
        
        Returns:
            True if collection was deleted, False if it didn't exist.
        
        Design note:
            Deletion is idempotent: deleting a non-existent collection returns False
            without error, making this safe for cleanup operations.
        """
        if collection_name is None:
            collection_name = self.collection_name
        
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")
            return False


def build_qdrant_collection(
    chunks: list,
    collection_name: str,
    qdrant_url: str,
    openai_api_key: str,
    model_name: str = "text-embedding-3-small",
) -> DenseRetriever:
    """
    Convenience function to create a retriever and index chunks in one call.
    
    This factory function combines DenseRetriever initialization and indexing,
    providing a simple entry point for common use cases. Use this when you want
    to go from raw chunks to a searchable collection in one operation.
    
    Args:
        chunks: List of chunk objects to index (must have .text/.text_with_heading
               and .metadata attributes).
        collection_name: Name for the Qdrant collection.
        qdrant_url: URL to Qdrant instance.
        openai_api_key: OpenAI API key for embeddings.
        model_name: OpenAI embedding model. Defaults to 'text-embedding-3-small'.
    
    Returns:
        Initialized DenseRetriever with all chunks indexed and ready for search.
    
    Example:
        >>> retriever = build_qdrant_collection(
        ...     chunks=my_chunks,
        ...     collection_name='financial_docs',
        ...     qdrant_url='http://localhost:6333',
        ...     openai_api_key='sk-...',
        ... )
        >>> results = retriever.search('What was the revenue?')
    """
    retriever = DenseRetriever(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        openai_api_key=openai_api_key,
        model_name=model_name,
    )
    
    count = retriever.index_chunks(chunks, collection_name=collection_name)
    logger.info(f"Built Qdrant collection '{collection_name}' with {count} chunks")
    
    return retriever
