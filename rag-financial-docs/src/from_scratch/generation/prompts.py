"""
Prompt templates and utilities for RAG-based financial document Q&A system.

This module provides standardized prompts that enforce faithful, grounded responses
by requiring the model to:
1. Answer ONLY from provided context chunks
2. Cite sources by chunk index
3. Explicitly state when information is not available
4. Never hallucinate or invent information

Design Philosophy:
- Numbered citations ([1], [2], etc.) enable precise source tracking and user verification
- Mandatory "I don't know" responses prevent hallucinations and protect against liability
- Explicit context formatting reduces cognitive load on the LLM
- Zero temperature in generation ensures deterministic, consistent answers
"""

from typing import List, Dict, Any


# System prompt that enforces grounded, factual responses
SYSTEM_PROMPT = """You are a financial document analyst assistant. Your role is to answer questions about financial documents using ONLY the context provided below.

CRITICAL RULES:
1. Answer ONLY from the provided context chunks. Do not use any external knowledge.
2. When you reference information, cite the source by chunk index (e.g., "According to [1], ...").
3. If the answer is not in the provided context, respond with: "I don't know - this information is not available in the provided documents."
4. Never hallucinate, invent, or speculate about information not explicitly stated in the context.
5. If multiple chunks are relevant, cite all of them.
6. Be precise and avoid generalizations beyond what the documents state.

Your responses should be helpful, accurate, and always traceable to the source documents."""


# RAG prompt template that structures context and question
RAG_PROMPT_TEMPLATE = """Context from financial documents:

{context}

Question: {question}

Please answer the question using ONLY the context provided above. Remember to cite chunks by index."""


def build_context_block(chunks: List[Any]) -> str:
    """
    Format retrieved chunks into numbered context blocks for LLM consumption.
    
    This function transforms raw chunk objects into a standardized format that:
    - Is easy for LLMs to parse and reference
    - Preserves source metadata for traceability
    - Maintains chunk order for consistent numbering
    - Prevents ambiguity about chunk boundaries
    
    Args:
        chunks: List of chunk objects with .text and .metadata attributes.
                Each chunk should have metadata with keys: 'doc_type', 'source_file', 'page'.
                Example metadata:
                {
                    'doc_type': 'Annual Report',
                    'source_file': 'AAPL_10K_2024.pdf',
                    'page': 42
                }
    
    Returns:
        Formatted string with numbered context blocks, suitable for inclusion in prompts.
        Format example:
        
        [1] Source: Annual Report, AAPL_10K_2024.pdf, page 42
        Apple generated revenue of $383.3 billion in fiscal 2024, representing growth...
        
        [2] Source: Quarterly Report, AAPL_Q1_2024.pdf, page 5
        The gross margin improved to 46.2% due to product mix...
    
    Design Decision - Numbered Format:
    - Sequential numbering ([1], [2], etc.) enables precise citations in responses
    - Chunk numbers serve as stable references throughout the Q&A interaction
    - Users can verify claims by looking up the corresponding chunk
    - LLM can reliably reference citations even if paraphrasing content
    
    Design Decision - Metadata Preservation:
    - Document type helps users understand source credibility (10-K vs. press release)
    - Source filename enables document retrieval for full context
    - Page numbers allow exact pinpointing in original PDF
    """
    if not chunks:
        return "[No context chunks provided]"
    
    formatted_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        # Extract metadata with safe defaults
        metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
        doc_type = metadata.get('doc_type', 'Unknown')
        source_file = metadata.get('source_file', 'Unknown')
        page = metadata.get('page', 'Unknown')
        
        # Get chunk text
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        # Format as numbered block
        header = f"[{idx}] Source: {doc_type}, {source_file}, page {page}"
        formatted_blocks.append(f"{header}\n{text}")
    
    return "\n\n".join(formatted_blocks)


def build_prompt(question: str, chunks: List[Any]) -> str:
    """
    Construct the complete prompt for the LLM by combining system context and question.
    
    This function orchestrates the full prompt assembly, ensuring that:
    - Retrieved chunks are properly formatted and numbered
    - The question is clearly separated from context
    - All necessary instructions are present
    - The format is optimized for accurate retrieval and generation
    
    Args:
        question: The user's question about the financial documents.
        chunks: List of retrieved chunks most relevant to the question.
    
    Returns:
        Complete prompt string ready to send to the LLM, combining:
        - Context blocks formatted by build_context_block()
        - The user's question
        - RAG_PROMPT_TEMPLATE structure
    
    Design Decision - Separation of Concerns:
    - Chunking logic isolated in build_context_block() for reusability
    - Template-based prompt construction allows easy modifications
    - Clear separation between context and question reduces confusion
    
    Design Decision - Prompt Structure Order:
    - Context comes BEFORE question to priming the model with relevant information
    - Question at the end focuses attention on what needs answering
    - This ordering improves relevance of citations
    
    Example:
        >>> chunks = [ChunkA, ChunkB]
        >>> prompt = build_prompt("What was Q1 revenue?", chunks)
        >>> # Prompt will include numbered chunks [1], [2] followed by question
    """
    context_block = build_context_block(chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_block, question=question)
    return prompt
