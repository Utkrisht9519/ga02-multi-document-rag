from groq import Groq
import streamlit as st


MAX_CONTEXT_CHARS = 12000  # Safe limit for llama3-8b


def truncate_context(text, max_chars=MAX_CONTEXT_CHARS):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Context truncated]"


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
    web_sources = []

    # ---------------- Document Retrieval ----------------
    if vectorstore and (not use_web or use_hybrid):
        docs = vectorstore.similarity_search(question, k=3)
        for d in docs:
            context_blocks.append(d.page_content)
            if "source" in d.metadata:
                document_sources.add(d.metadata["source"])

    # ---------------- Web Retrieval ----------------
    if web_context and (use_web or use_hybrid):
        context_blocks.append(web_context.get("content", ""))
        web_sources = web_context.get("sources", [])

    # ---------------- Validation ----------------
    if not context_blocks:
        return (
            "⚠️ No context available to answer this question. "
            "Please upload documents or enable web search.",
            {
                "document_sources": [],
                "web_sources": []
            }
        )

    combined_context = "\n\n".join(context_blocks)
    combined_context = truncate_context(combined_context)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.
If the answer is not present, say so clearly.

Context:
{combined_context}

Question:
{question}
"""

    # ---------------- Safe Groq Call ----------------
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = completion.choices[0].message.content

        return answer, {
            "document_sources": sorted(document_sources),
            "web_sources": web_sources,
        }

    except Exception as e:
        # ✅ Prevent app crash
        return (
            "❌ The AI model could not process this request due to context limits. "
            "Try asking a more specific question or uploading fewer documents.",
            {
                "document_sources": sorted(document_sources),
                "web_sources": web_sources,
            }
        )
