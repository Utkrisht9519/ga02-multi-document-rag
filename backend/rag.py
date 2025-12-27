from groq import Groq
import streamlit as st


def generate_answer(
    question,
    vectorstore=None,
    web_context=None,
    use_web=False,
    use_hybrid=False
):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    context_blocks = []
    document_sources = set()

    # Document Retrieval
    if vectorstore and (not use_web or use_hybrid):
        docs = vectorstore.similarity_search(question, k=4)
        for d in docs:
            context_blocks.append(d.page_content)
            if "source" in d.metadata:
                document_sources.add(d.metadata["source"])

    # Web Retrieval
    web_sources = []
    if web_context and (use_web or use_hybrid):
        context_blocks.append(web_context["content"])
        web_sources = web_context.get("sources", [])

    prompt = f"""
Use the context below to answer the question clearly and concisely.

Context:
{"\n\n".join(context_blocks)}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    answer = completion.choices[0].message.content

    return answer, {
        "document_sources": sorted(document_sources),
        "web_sources": web_sources
    }
