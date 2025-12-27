from typing import List, Tuple
from langchain.schema import Document
from groq import Groq
import streamlit as st

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def generate_answer(
    question: str,
    documents: List[Document],
    web_context: List[dict] | None,
    mode: str,  # "doc", "web", "hybrid"
) -> Tuple[str, dict]:
    """
    Returns:
        answer (str)
        sources (dict):
            {
                "doc_citations": [str],
                "doc_files": [str],
                "web_sources": [str]
            }
    """

    doc_text = ""
    doc_citations = set()
    doc_files = set()

    if documents:
        for doc in documents:
            doc_text += doc.page_content + "\n\n"
            if "source" in doc.metadata:
                doc_citations.add(doc.metadata["source"])
            if "file_name" in doc.metadata:
                doc_files.add(doc.metadata["file_name"])

    web_text = ""
    web_sources = []

    if web_context:
        for item in web_context:
            web_text += f"{item['content']}\n\n"
            web_sources.append(item["source"])

    prompt = f"""
You are a RAG-based assistant.

QUESTION:
{question}

DOCUMENT CONTEXT:
{doc_text if mode in ["doc", "hybrid"] else "N/A"}

WEB CONTEXT:
{web_text if mode in ["web", "hybrid"] else "N/A"}

RULES:
- Answer only from the provided context
- Be concise and factual
- Do NOT hallucinate
"""

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    sources = {
        "doc_citations": sorted(doc_citations),
        "doc_files": sorted(doc_files),
        "web_sources": web_sources,
    }

    return answer, sources
