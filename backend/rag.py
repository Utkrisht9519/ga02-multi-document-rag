import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def generate_answer(question, vectorstore, web_context="", hybrid=False):
    docs = vectorstore.similarity_search(question, k=4)

    doc_context = "\n\n".join(d.page_content for d in docs)

    combined_context = doc_context
    if hybrid and web_context:
        combined_context += f"\n\nWeb Context:\n{web_context}"

    prompt = f"""
Answer the question using the context below.
Always cite sources at the end.

Context:
{combined_context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    sources = list(
        set(
            d.metadata.get("source", "Uploaded document")
            for d in docs
        )
    )

    return answer, sources
