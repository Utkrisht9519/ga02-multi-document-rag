import os
from groq import Groq

# Groq Client Setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
        context = f"""
DOCUMENT CONTEXT:
{doc_context}

WEB CONTEXT:
{web_context}
"""
    else:
        context = f"""
DOCUMENT CONTEXT:
{doc_context}
"""

    # Prompt
    prompt = f"""
You are a professional research assistant.

Rules:
- Use ONLY the provided context
- Summarize in your own words
- Be accurate and concise
- Do NOT copy verbatim text
- Do NOT hallucinate

Context:
{context}

Question:
{question}

Answer:
"""

    # Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
