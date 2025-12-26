import streamlit as st
from pathlib import Path
import tempfile

from backend.ingestion import load_documents
from backend.chunking import chunk_documents
from backend.vector_store import create_vector_store
from backend.web_search import tavily_search
from backend.rag import generate_answer


# Page Config
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    layout="wide",
)


# Sidebar
st.sidebar.title("📘 About")

st.sidebar.markdown(
    """
**Multi-Document RAG Chatbot can:**
- Answer questions from uploaded documents
- Search the web using Tavily
- Combine document + web context intelligently
"""
)

enable_web = st.sidebar.toggle("🌐 Enable Web Search", value=False)
enable_hybrid = st.sidebar.toggle(
    "🔀 Enable Hybrid Search (Docs + Web)",
    value=False,
    disabled=not enable_web,
)

st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "📄 Upload Documents (PDF / TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)


# Main UI
st.title("📚 Multi-Document RAG Chatbot")

question = st.text_input("Ask a question about your documents")


# Ask Button
if st.button("Ask"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Save uploaded files temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for file in uploaded_files:
            file_path = tmp_path / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())

        # Load & process documents
        with st.spinner("Processing documents..."):
            documents = load_documents(tmp_path)
            chunks = chunk_documents(documents)
            vectorstore = create_vector_store(chunks)

        # Optional web search
        web_context = None
        if enable_web:
            with st.spinner("Searching the web..."):
                web_context = tavily_search(question)

        # Generate answer
        with st.spinner("Generating answer..."):
            result = generate_answer(
                question=question,
                vectorstore=vectorstore,
                use_web=enable_hybrid,
                web_context=web_context,
            )

        # Display Answer
        st.subheader("Answer")
        st.write(result["answer"])

        # Evidence & Sources
        st.subheader("Evidence & Sources")

        for src in result["sources"]:
            with st.expander(
                f"Source [{src['id']}] — {src['source']} (page {src['page']})"
            ):
                st.write(src["content"])


# Footer
st.divider()
st.caption(
    "Built with Streamlit • LangChain • FAISS • Groq • Tavily"
)
