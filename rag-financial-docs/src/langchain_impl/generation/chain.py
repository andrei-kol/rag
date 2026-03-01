"""
LangChain RAG Chain Implementation

This module demonstrates how LangChain's Expression Language (LCEL) abstracts
the generation pipeline we built from scratch.

From-scratch vs. LangChain:
- From-scratch: Manual orchestration
  1. Query embedding → 2. Vectorstore search → 3. Result formatting → 
  4. Prompt construction → 5. LLM call → 6. Output parsing
  All done sequentially with explicit Python function calls and state passing.

- LangChain LCEL: Declarative composition
  Chains are built as composable Runnables using the pipe operator (|).
  This enables streaming, parallelization, debugging via LangSmith,
  and deployment without code changes.
"""

import operator
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


RAG_SYSTEM_PROMPT = """You are a financial analyst assistant specializing in document analysis and research.

When answering questions, you MUST:
1. Base your answer ONLY on the provided documents and context
2. Cite specific sources (document name, page, section) for every claim
3. If information is not in the documents, explicitly state "This information is not in the provided documents"
4. Highlight any limitations or conflicting information between sources
5. Provide numerical data with context (e.g., "Revenue grew from $X in 2023 to $Y in 2024")

Format citations as: [Document Name, Page X] or [Source: Company Report, Q4 2024]

Never:
- Make up or infer information not in the documents
- Assume context beyond what is explicitly stated
- Provide financial advice or recommendations
- Extrapolate future trends without explicit basis in the documents"""


def format_docs(docs: List[Document]) -> str:
    """
    Format LangChain Documents into a numbered context block for the prompt.

    In our from-scratch implementation, we manually joined document text with
    separators. This function standardizes that formatting and adds metadata
    context (source, page number) so the LLM can cite sources accurately.

    Args:
        docs: List of LangChain Document objects from retriever

    Returns:
        Formatted string with numbered documents and metadata
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        # Extract metadata for citation
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        heading = doc.metadata.get("heading", "")

        # Build citation prefix
        citation = f"[Document {i}: {source}, Page {page}]"
        if heading:
            citation += f" Section: {heading}"

        # Format content
        formatted.append(f"{citation}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def build_rag_chain(
    retriever: BaseRetriever,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Runnable:
    """
    Build a RAG chain using LangChain Expression Language (LCEL).

    This demonstrates the power of LCEL composability. The chain is defined
    as a series of pipes (|), making the data flow explicit and testable.

    Pipeline:
    1. retriever: Takes {"question": "..."} and retrieves relevant Documents
    2. format_docs: Converts Document list to formatted string with citations
    3. prompt: Injects formatted docs into the ChatPromptTemplate
    4. llm: Calls ChatOpenAI with the filled prompt
    5. parser: Extracts string response from LLM output

    In our from-scratch version, this would be:
    ```python
    def rag_generate(question):
        # Step 1: Embed and search
        query_embedding = embed(question)
        docs = vectorstore.search(query_embedding, top_k=5)
        
        # Step 2-3: Format
        context = "\\n\\n".join([d.text for d in docs])
        
        # Step 4-5: Prompt and generate
        prompt_text = RAG_SYSTEM_PROMPT + "\\nContext:\\n" + context + f"\\nQuestion: {question}"
        response = llm.generate(prompt_text)
        return response
    ```

    LCEL advantages:
    - Streaming: Chain can be called with .stream() for token-by-token output
    - Debugging: LangSmith integration traces each step
    - Parallelization: Multiple branches can execute in parallel (see build_rag_chain_with_sources)
    - Composability: Chains can be nested and reused

    Args:
        retriever: LangChain retriever (from get_retriever)
        openai_api_key: OpenAI API key
        model: Model name (default: gpt-4o-mini for cost efficiency)
        temperature: Sampling temperature (0.0 = deterministic for RAG)

    Returns:
        Runnable: LCEL chain that takes {"question": str} and returns str

    Example:
        >>> chain = build_rag_chain(retriever, openai_api_key)
        >>> result = chain.invoke({"question": "What was the revenue in Q4?"})
        >>> print(result)
        # Output: "Based on [Document 1: 2024 Annual Report, Page 15], ..."
    """
    # Create prompt template with variables for context and question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            (
                "human",
                """Based on the following documents, answer the question.

Documents:
{context}

Question: {question}""",
            ),
        ]
    )

    # Initialize LLM with low temperature for factual consistency
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=openai_api_key)

    # Build LCEL chain with explicit pipe operations
    # Each pipe (|) passes output of left side as input to right side

    # Step 1: Retrieve documents
    # Input: {"question": "..."}
    # Output: List[Document]
    retrieval_step = retriever

    # Step 2: Format documents
    # Input: List[Document]
    # Output: str (formatted context with citations)
    format_step = RunnableLambda(format_docs)

    # Step 3-5: Combine retrieval with prompt, LLM, and parser
    # RunnablePassthrough keeps {"question": "..."} in the chain
    # while the formatted context flows to the "context" variable
    chain = (
        {"context": retrieval_step | format_step, "question": RunnablePassthrough()}
        | prompt  # Fills template with context and question
        | llm  # Calls ChatOpenAI
        | StrOutputParser()  # Extracts string from LLM response
    )

    return chain


def build_rag_chain_with_sources(
    retriever: BaseRetriever,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
) -> Runnable:
    """
    Build a RAG chain that returns both answer and source documents.

    This demonstrates LCEL's RunnablePassthrough.assign() pattern, which
    allows branching and parallel execution. The answer generation runs
    independently from the retrieval, and both results are combined.

    In our from-scratch version, we'd return both manually:
    ```python
    def rag_with_sources(question):
        docs = search(question)
        answer = generate(docs, question)
        return {"answer": answer, "sources": docs}
    ```

    With LCEL, this is more elegant:
    - The chain explicitly shows that we retrieve documents
    - The retrieved documents flow to both the answer generation AND the output
    - No manual variable passing or state management

    Args:
        retriever: LangChain retriever
        openai_api_key: OpenAI API key
        model: Model name (default: gpt-4o-mini)

    Returns:
        Runnable: Chain that takes {"question": str} and returns 
                 {"answer": str, "sources": List[Document]}

    Example:
        >>> chain = build_rag_chain_with_sources(retriever, openai_api_key)
        >>> result = chain.invoke({"question": "What was the revenue?"})
        >>> print(result["answer"])
        >>> print(f"Sources: {[doc.metadata['source'] for doc in result['sources']]}")
    """
    # Create the base answer chain (same as build_rag_chain)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            (
                "human",
                """Based on the following documents, answer the question.

Documents:
{context}

Question: {question}""",
            ),
        ]
    )

    llm = ChatOpenAI(model=model, temperature=0.0, api_key=openai_api_key)

    # Build answer chain
    answer_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Build chain that returns both answer and sources
    # RunnablePassthrough.assign() keeps the original input and adds new keys
    chain = (
        {"sources": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=answer_chain)
        | RunnableLambda(lambda x: {"answer": x["answer"], "sources": x["sources"]})
    )

    return chain
