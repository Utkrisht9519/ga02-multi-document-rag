import streamlit as st

from backend.ingestion import ingest_uploaded_files
from backend.vector_store import create_vector_store
from backend.rag import generate_answer
from backend.web_search import tavily_search


st.set_page_config(page_title="Multi-Document RAG Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📘 About")

    st.markdown("""
    **Multi-Document RAG Chatbot can:**
    - Answer questions from uploaded documents
    - Search the web using Tavily
    - Combine document + web context intelligently
    """)

    use_web = st.toggle("🌐 Enable Web Search", value=False)
    use_hybrid = st.toggle("🔀 Enable Hybrid Search", value=False)

    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

# Main UI
st.title("Multi-Document RAG Chatbot")

question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. Summarize the key findings of the uploaded papers"
)

if st.button("Ask") and question:
    if uploaded_files:
        docs, uploaded_filenames = ingest_uploaded_files(uploaded_files)
        vectorstore = create_vector_store(docs)
    else:
        vectorstore = None
        uploaded_filenames = []

    web_context = None
    if use_web or use_hybrid:
        web_context = tavily_search(question)

    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(
            question=question,
            vectorstore=vectorstore,
            web_context=web_context,
            use_web=use_web,
            use_hybrid=use_hybrid
        )

    st.markdown("## ✅ Answer")
    st.write(answer)

    # Sources Rendering Logic
    st.markdown("### 📚 Sources")

    if use_web and not use_hybrid:
        # Web-only
        for src in sources["web_sources"]:
            st.markdown(f"- {src}")

    elif use_hybrid:
        # Hybrid
        if sources["web_sources"]:
            st.markdown("**Web Sources:**")
            for src in sources["web_sources"]:
                st.markdown(f"- {src}")

        if sources["document_sources"]:
            st.markdown("**Uploaded Documents:**")
            for doc in sources["document_sources"]:
                st.markdown(f"- {doc}")

    else:
        # Document-only
        for doc in sources["document_sources"]:
            st.markdown(f"- {doc}")
