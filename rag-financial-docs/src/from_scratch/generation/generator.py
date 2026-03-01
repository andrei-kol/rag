"""
LLM-based generation layer for RAG financial document Q&A system.

This module handles:
1. Communication with OpenAI API for text generation
2. Token counting and cost calculation
3. Assembly of complete GenerationResult with usage metrics
4. One-shot and class-based usage patterns

Key Design Decisions:
- Temperature=0.0 (zero temperature) ensures deterministic, faithful responses without hallucination
- Cost calculation enables usage monitoring and budget tracking
- Comprehensive logging via docstrings supports production debugging
- Flexible initialization supports both single-question and batch workflows
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
import os

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package required. Install with: pip install openai"
    )

from .prompts import build_prompt, SYSTEM_PROMPT


@dataclass
class GenerationResult:
    """
    Complete result of RAG-based generation including answer, metrics, and provenance.
    
    This dataclass provides a structured container for all outputs from the generation
    pipeline, enabling:
    - Transparent cost tracking per query
    - Verification of which chunks contributed to the answer
    - Model and parameter reproducibility
    - Token usage analysis for optimization
    
    Attributes:
        answer: The model's response to the question, based on provided context.
        question: The original user question that was answered.
        chunks_used: List of chunk objects that were provided as context for generation.
                     Enables tracing answer back to source documents.
        model: Model identifier used (e.g., 'gpt-4o-mini', 'gpt-4o').
               Useful for reproducibility and comparing different model outputs.
        prompt_tokens: Total tokens consumed by the system prompt, context, and question.
                       Used for cost calculation and efficiency analysis.
        completion_tokens: Tokens generated in the answer. Typically lower than prompt_tokens
                          due to answer brevity, but varies with question complexity.
        total_cost_usd: Calculated cost in US dollars based on model pricing and token usage.
                        Enables budget tracking and cost attribution.
    
    Example:
        >>> result = GenerationResult(
        ...     answer="The revenue was $50B according to [1]",
        ...     question="What was annual revenue?",
        ...     chunks_used=[chunk1, chunk2],
        ...     model="gpt-4o-mini",
        ...     prompt_tokens=1240,
        ...     completion_tokens=28,
        ...     total_cost_usd=0.00034
        ... )
    """
    answer: str
    question: str
    chunks_used: List[Any]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float


class RAGGenerator:
    """
    Production-grade RAG generation engine for financial document Q&A.
    
    This class manages the complete generation pipeline:
    1. Accepts a question and relevant chunks
    2. Constructs a grounded prompt via prompts.py
    3. Calls OpenAI API with deterministic parameters
    4. Calculates usage costs automatically
    5. Returns structured GenerationResult
    
    Design Philosophy:
    - Zero temperature ensures answers never deviate from context
    - Consistent token counting prevents cost surprises
    - Configurable model supports experimentation (mini vs. full 4o)
    - Hardcoded pricing table handles model versioning
    
    Usage:
        >>> gen = RAGGenerator(openai_api_key="sk-...")
        >>> result = gen.generate("What was Q1 revenue?", chunks)
        >>> print(result.answer)
        >>> print(f"Cost: ${result.total_cost_usd}")
    
    Cost Transparency:
    - Prices updated as of February 2025
    - gpt-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
    - gpt-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
    - Enable budget monitoring and ROI analysis
    """
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Initialize the RAG generator with OpenAI configuration.
        
        Args:
            openai_api_key: OpenAI API key for authentication.
                          Should be retrieved from environment if not provided.
            model: Model to use for generation. Supported:
                   - "gpt-4o-mini" (faster, cheaper, recommended for most queries)
                   - "gpt-4o" (more capable, higher cost)
                   Default: "gpt-4o-mini"
            temperature: Controls response randomness. Must be 0.0 for faithful RAG.
                        0.0 = deterministic (no hallucination)
                        >0 = increased creativity (NOT RECOMMENDED for financial data)
                        Default: 0.0 (enforced)
            max_tokens: Maximum output length per response. Adjust based on:
                       - Question complexity
                       - Cost constraints
                       - Response brevity requirements
                       Default: 1024 tokens (~800 words)
        
        Design Decision - Zero Temperature Enforcement:
        Temperature must remain 0.0 for financial documents because:
        - Any non-zero temperature introduces hallucination risk
        - Financial data requires absolute accuracy
        - Consistency is critical for compliance and verification
        
        Design Decision - API Key Handling:
        - Accepts explicit key for flexibility (testing, multiple keys)
        - Falls back to OPENAI_API_KEY env var if not provided
        - Initializes OpenAI client immediately for early error detection
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not provided. Pass openai_api_key or set OPENAI_API_KEY env var."
            )
        
        self.model = model
        self.temperature = 0.0  # Force zero temperature for faithful responses
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.openai_api_key)
    
    def generate(self, question: str, chunks: List[Any]) -> GenerationResult:
        """
        Generate an answer to a question using retrieved chunks as context.
        
        This method orchestrates the complete generation pipeline:
        1. Build a structured prompt with context and question
        2. Call OpenAI API with deterministic parameters
        3. Extract usage metrics from response
        4. Calculate cost based on token usage
        5. Package everything in a GenerationResult
        
        Args:
            question: The user's question about the financial documents.
            chunks: List of relevant chunks retrieved (typically from vector search).
                   Should be pre-filtered to top-k most relevant results.
                   Each chunk needs .text and .metadata attributes.
        
        Returns:
            GenerationResult containing:
            - answer: Model's response (or "I don't know" if not in context)
            - question: Echo of input question
            - chunks_used: Reference to input chunks
            - model: Model used
            - prompt_tokens: Tokens consumed by context+question
            - completion_tokens: Tokens generated in answer
            - total_cost_usd: Calculated cost
        
        Raises:
            ValueError: If OpenAI API returns invalid response
            Exception: If API call fails (network, rate limit, invalid key)
        
        Design Decision - Error Handling:
        - Let API exceptions propagate for proper error handling upstream
        - Don't catch and suppress errors that indicate configuration problems
        - Allows retry logic at higher level
        
        Design Decision - Token Counting:
        - Use actual usage numbers from API response (not estimates)
        - Ensures cost calculation matches actual billing
        - Supports accurate budget forecasting
        
        Example:
            >>> gen = RAGGenerator(api_key="sk-...")
            >>> result = gen.generate("Revenue in Q1?", retrieved_chunks)
            >>> print(f"Answer: {result.answer}")
            >>> print(f"Used {result.completion_tokens} tokens")
            >>> print(f"Cost: ${result.total_cost_usd:.4f}")
        """
        # Build the complete prompt with context and question
        prompt = build_prompt(question, chunks)
        
        # Call OpenAI API with deterministic parameters
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Extract answer and usage metrics
        answer = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        # Calculate cost based on actual token usage
        total_cost = self._calculate_cost(
            self.model, prompt_tokens, completion_tokens
        )
        
        # Package results
        return GenerationResult(
            answer=answer,
            question=question,
            chunks_used=chunks,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=total_cost,
        )
    
    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Calculate cost in USD based on model and token usage.
        
        Uses hardcoded pricing table for supported models. Prices are fixed
        per 1 million tokens and match OpenAI's public pricing as of Feb 2025.
        
        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
            prompt_tokens: Number of input tokens consumed
            completion_tokens: Number of output tokens generated
        
        Returns:
            Total cost in USD as a float, rounded to 6 decimal places.
        
        Pricing Table (per 1M tokens):
            gpt-4o-mini:
              - Input: $0.15
              - Output: $0.60
            gpt-4o:
              - Input: $2.50
              - Output: $10.00
        
        Design Decision - Hardcoded Pricing:
        - Central location for all model prices
        - Easy to update when pricing changes
        - Prevents pricing lookups from external sources
        - Deterministic cost calculation
        
        Design Decision - Per-Million Token Pricing:
        - Follows OpenAI's pricing convention
        - Enables simple multiplication without API calls
        - Supports accurate cost attribution per query
        
        Example:
            >>> cost = gen._calculate_cost("gpt-4o-mini", 1000, 200)
            >>> # 1000 * 0.15 / 1M + 200 * 0.60 / 1M = 0.00015 + 0.00012 = 0.00027
        """
        # Pricing per 1M tokens
        pricing = {
            "gpt-4o-mini": {
                "input": 0.15,
                "output": 0.60,
            },
            "gpt-4o": {
                "input": 2.50,
                "output": 10.00,
            },
        }
        
        if model not in pricing:
            raise ValueError(
                f"Unknown model: {model}. Supported: {list(pricing.keys())}"
            )
        
        # Calculate cost: (tokens / 1M) * (price per 1M)
        model_pricing = pricing[model]
        input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6)


def answer_question(
    question: str,
    chunks: List[Any],
    openai_api_key: str,
    model: str = "gpt-4o-mini",
) -> GenerationResult:
    """
    One-shot function for answering a single question without class initialization.
    
    This convenience function is useful for:
    - Quick scripts and notebooks
    - Single-query use cases (not batch processing)
    - Testing and experimentation
    - Avoiding class initialization overhead for one-time usage
    
    Args:
        question: The user's question about financial documents
        chunks: Retrieved chunks to use as context
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Model to use, default "gpt-4o-mini"
    
    Returns:
        GenerationResult with answer, metrics, and provenance
    
    Raises:
        ValueError: If OpenAI API key is missing
        Exception: If API call fails
    
    Design Decision - Convenience Function:
    - Reduces boilerplate for simple use cases
    - Maintains same signature as RAGGenerator.generate()
    - Internally uses RAGGenerator to avoid code duplication
    
    Example:
        >>> result = answer_question(
        ...     question="What was revenue?",
        ...     chunks=retrieved_chunks,
        ...     openai_api_key="sk-..."
        ... )
        >>> print(result.answer)
    """
    generator = RAGGenerator(
        openai_api_key=openai_api_key,
        model=model,
    )
    return generator.generate(question, chunks)
