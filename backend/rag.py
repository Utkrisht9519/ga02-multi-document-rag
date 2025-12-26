from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os

# LLM Setup (Groq – Free Tier)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
)


# Answer Generation (Hybrid RAG)
def generate_answer(
    question: str,
    vectorstore,
    web_context: str | None = None,
    top_k: int = 4,
) -> str:
    """
    Generate an answer using:
    - FAISS vector search (documents)
    - Optional Tavily web search context

    Args:
        question (str): User question
        vectorstore: FAISS vector store
        web_context (str): Web search results (optional)
        top_k (int): Number of document chunks to retrieve

    Returns:
        str: Final answer
    """

    # Retrieve document chunks
    docs = vectorstore.similarity_search(question, k=top_k)

    if not docs:
        doc_context = "No relevant document content found."
    else:
        doc_context = "\n\n".join(
            [f"[Document]\n{doc.page_content}" for doc in docs]
        )

    # Combine with web context
    if web_context:
        combined_context = f"""
DOCUMENT CONTEXT:
{doc_context}

WEB CONTEXT:
{web_context}
"""
    else:
        combined_context = f"""
DOCUMENT CONTEXT:
{doc_context}
"""

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a professional research assistant.

Your task:
- Read the context carefully
- Summarize information in your own words
- Provide a clear, concise, accurate answer
- Do NOT copy text verbatim
- Do NOT hallucinate
- If information is missing, clearly say so

Context:
{context}

Question:
{question}

Answer:
""",
    )

    # Generate Answer
    response = llm.invoke(
        prompt.format(context=combined_context, question=question)
    )

    return response.content
