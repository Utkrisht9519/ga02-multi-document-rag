from groq import Groq
import streamlit as st

# Groq Client Setup
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Generate Answer Function
def generate_answer(
    question: str,
    vectorstore,
    use_web: bool = False,
    web_context: str | None = None,
    top_k: int = 4,
):
    """
    Generates an answer using document RAG and optional web context.
    Returns answer + sources for citations.
    """

    # Retrieve document chunks
    docs = vectorstore.similarity_search(question, k=top_k)

    sources = []
    context_blocks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Uploaded document")
        page = doc.metadata.get("page", "N/A")

        sources.append(
            {
                "id": i,
                "source": source,
                "page": page,
                "content": doc.page_content,
            }
        )

        context_blocks.append(f"[{i}] {doc.page_content}")

    document_context = "\n\n".join(context_blocks)

    # Combine with web context (Hybrid Search)
    if use_web and web_context:
        final_context = f"""
DOCUMENT CONTEXT:
{document_context}

WEB CONTEXT:
{web_context}
"""
    else:
        final_context = f"""
DOCUMENT CONTEXT:
{document_context}
"""

    # Prompt (forces citations)
    prompt = f"""
You are a factual research assistant.

Rules:
- Answer ONLY from the given context
- Cite sources using [1], [2], etc.
- Do NOT hallucinate
- If the answer is not found, say "Not found in provided documents."

Context:
{final_context}

Question:
{question}

Answer (with citations):
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }
