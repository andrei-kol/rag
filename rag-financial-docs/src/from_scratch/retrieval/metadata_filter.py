"""
Metadata filtering for retrieval: search within a subset based on document properties.

WHAT IS METADATA FILTERING?
    Metadata filtering is like a SQL WHERE clause applied before vector search.
    Instead of searching the entire document corpus, filter to a subset based on:
    - Client ID (multi-tenant isolation)
    - Document type (annual reports, contracts, risk assessments)
    - Year or date range (recent data, specific periods)
    - Confidentiality level (public vs confidential)
    - Source file (specific document)
    
    Benefits:
    - Reduces search space (faster queries)
    - Improves precision (fewer false positives from irrelevant docs)
    - Enables compliance (confidentiality controls)
    - Supports multi-tenancy (data isolation)

WHEN TO USE:
    - "Show me annual reports from 2022-2024 for client ABC"
      → Filter: doc_type="annual_report", year_range=(2022, 2024), client_id="ABC"
    
    - "Search public financial information only"
      → Filter: confidentiality="public"
    
    - "Recent contracts mentioning risk"
      → Filter: doc_type="contract", year=2024
    
    - Multi-tenant SaaS: always filter by client_id before search

IMPLEMENTATION BACKENDS:
    - Qdrant: to_qdrant_filter() converts to native Qdrant Filter objects
    - In-memory: apply_filter_to_list() filters Python lists (testing, fallback)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client.models import Filter


@dataclass
class MetadataFilter:
    """
    Filters for document metadata.
    
    All fields are optional (None means no filter on that field).
    When multiple fields are set, all conditions are AND-ed together.
    
    Attributes:
        client_id: Filter by client ID (for multi-tenant systems)
        doc_type: Document type (e.g., "annual_report", "contract", "press_release")
        year: Exact year match
        year_range: Tuple of (start_year, end_year) inclusive
        confidentiality: Confidentiality level (e.g., "public", "confidential", "restricted")
        source_file: Exact filename match (e.g., "2024-Q3-earnings.pdf")
    """
    
    client_id: str | None = None
    doc_type: str | None = None
    year: int | None = None
    year_range: tuple[int, int] | None = None
    confidentiality: str | None = None
    source_file: str | None = None
    
    def is_empty(self) -> bool:
        """Check if all filter fields are None."""
        return all(
            getattr(self, field) is None
            for field in [
                "client_id",
                "doc_type",
                "year",
                "year_range",
                "confidentiality",
                "source_file",
            ]
        )


def to_qdrant_filter(mf: MetadataFilter) -> "Filter | None":
    """
    Convert MetadataFilter to Qdrant Filter object.
    
    Returns None if all filter fields are None (no filtering needed).
    
    Args:
        mf: MetadataFilter instance
        
    Returns:
        Qdrant Filter object or None if empty filter
        
    Raises:
        ImportError: If qdrant_client is not installed
        
    Example:
        mf = MetadataFilter(doc_type="annual_report", year=2024)
        qdrant_filter = to_qdrant_filter(mf)
        # Use with: client.search(query_vector, query_filter=qdrant_filter)
    """
    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        LogicalOperator,
    )
    
    if mf.is_empty():
        return None
    
    conditions: list = []
    
    # Client ID filter
    if mf.client_id is not None:
        conditions.append(
            FieldCondition(
                key="metadata.client_id",
                match=MatchValue(value=mf.client_id),
            )
        )
    
    # Document type filter
    if mf.doc_type is not None:
        conditions.append(
            FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value=mf.doc_type),
            )
        )
    
    # Exact year filter
    if mf.year is not None:
        conditions.append(
            FieldCondition(
                key="metadata.year",
                match=MatchValue(value=mf.year),
            )
        )
    
    # Year range filter
    if mf.year_range is not None:
        start_year, end_year = mf.year_range
        conditions.append(
            FieldCondition(
                key="metadata.year",
                range=Range(gte=start_year, lte=end_year),
            )
        )
    
    # Confidentiality filter
    if mf.confidentiality is not None:
        conditions.append(
            FieldCondition(
                key="metadata.confidentiality",
                match=MatchValue(value=mf.confidentiality),
            )
        )
    
    # Source file filter
    if mf.source_file is not None:
        conditions.append(
            FieldCondition(
                key="metadata.source_file",
                match=MatchValue(value=mf.source_file),
            )
        )
    
    # Combine all conditions with AND
    if not conditions:
        return None
    elif len(conditions) == 1:
        return Filter(must=conditions)
    else:
        return Filter(must=conditions)


def apply_filter_to_list(
    chunks: list,
    mf: MetadataFilter,
) -> list:
    """
    Pure Python filtering for document chunks.
    
    Useful for:
    - Unit tests (no external dependencies)
    - In-memory fallback when database unavailable
    - Local processing of document lists
    
    Args:
        chunks: List of objects with .metadata dict attribute
        mf: MetadataFilter instance
        
    Returns:
        Filtered list (same object type as input)
        
    Example:
        filtered = apply_filter_to_list(chunks, MetadataFilter(year=2024))
    """
    if mf.is_empty():
        return chunks
    
    def matches_filter(chunk) -> bool:
        """Check if chunk matches all filter conditions."""
        metadata = getattr(chunk, "metadata", {})
        
        # Client ID filter
        if mf.client_id is not None:
            if metadata.get("client_id") != mf.client_id:
                return False
        
        # Document type filter
        if mf.doc_type is not None:
            if metadata.get("doc_type") != mf.doc_type:
                return False
        
        # Exact year filter
        if mf.year is not None:
            if metadata.get("year") != mf.year:
                return False
        
        # Year range filter
        if mf.year_range is not None:
            chunk_year = metadata.get("year")
            if chunk_year is None:
                return False
            start_year, end_year = mf.year_range
            if not (start_year <= chunk_year <= end_year):
                return False
        
        # Confidentiality filter
        if mf.confidentiality is not None:
            if metadata.get("confidentiality") != mf.confidentiality:
                return False
        
        # Source file filter
        if mf.source_file is not None:
            if metadata.get("source_file") != mf.source_file:
                return False
        
        return True
    
    return [chunk for chunk in chunks if matches_filter(chunk)]


def public_only() -> MetadataFilter:
    """
    Factory for filtering to public documents only.
    
    Returns:
        MetadataFilter with confidentiality="public"
        
    Example:
        results = retriever.search(query, filter=public_only())
    """
    return MetadataFilter(confidentiality="public")


def by_client(client_id: str) -> MetadataFilter:
    """
    Factory for filtering by client ID.
    
    Useful for multi-tenant systems.
    
    Args:
        client_id: Client identifier
        
    Returns:
        MetadataFilter with specified client_id
        
    Example:
        results = retriever.search(query, filter=by_client("client_xyz"))
    """
    return MetadataFilter(client_id=client_id)


def by_doc_type(doc_type: str) -> MetadataFilter:
    """
    Factory for filtering by document type.
    
    Args:
        doc_type: Document type (e.g., "annual_report", "contract")
        
    Returns:
        MetadataFilter with specified doc_type
        
    Example:
        results = retriever.search(query, filter=by_doc_type("contract"))
    """
    return MetadataFilter(doc_type=doc_type)


def by_year(year: int) -> MetadataFilter:
    """
    Factory for filtering by exact year.
    
    Args:
        year: Year as integer (e.g., 2024)
        
    Returns:
        MetadataFilter with specified year
        
    Example:
        results = retriever.search(query, filter=by_year(2024))
    """
    return MetadataFilter(year=year)


def recent_years(n: int = 3) -> MetadataFilter:
    """
    Factory for filtering to recent years.
    
    Returns documents from the last n years (current year inclusive).
    
    Args:
        n: Number of recent years to include (default 3)
        
    Returns:
        MetadataFilter with year_range=(current_year - n, current_year)
        
    Example:
        # Include 2022, 2023, 2024 (if current year is 2024)
        results = retriever.search(query, filter=recent_years(3))
        
        # Include only 2024
        results = retriever.search(query, filter=recent_years(1))
    """
    current_year = datetime.now().year
    start_year = current_year - n + 1
    return MetadataFilter(year_range=(start_year, current_year))
