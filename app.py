import os
from pathlib import Path
import streamlit as st

from backend.ingestion import load_documents
from backend.chunking import chunk_documents
from backend.vector_store import create_vector_store
from backend.rag import generate_answer
from backend.web_search import web_search

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# -------------------------
# Constants
# -------------------------
UPLOAD_DIR = "documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("📘 About")

    st.markdown(
        """
        **Multi-Document RAG Chatbot** can:

        • Answer questions from uploaded documents  
        • Search the web using Tavily  
        • Combine document + web context intelligently  
        """
    )

    enable_web = st.toggle("🌐 Enable Web Search", value=False)

    st.divider()

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

# -------------------------
# Save uploaded files
# -------------------------
if uploaded_files:
    for file in uploaded_files:
        file_path = Path(UPLOAD_DIR) / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())

    st.sidebar.success("Documents uploaded successfully")

# -------------------------
# Main UI
# -------------------------
st.title("Multi-Document RAG Chatbot")
st.caption("Ask questions about your documents (and optionally the web)")

st.divider()

# -------------------------
# Question input (BOTTOM)
# -------------------------
question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What is the main idea of this document?"
)

ask_clicked = st.button("Ask")

# -------------------------
# RAG Pipeline
# -------------------------
if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Processing documents..."):
        docs = load_documents(UPLOAD_DIR)

        if not docs:
            st.warning("Please upload at least one document before asking a question.")
            st.stop()

        chunks = chunk_documents(docs)
        vectorstore = create_vector_store(chunks)

    # -------------------------
    # Web Search (optional)
    # -------------------------
    web_context = ""
    if enable_web:
        with st.spinner("Searching the web..."):
            web_context = web_search(question)

    # -------------------------
    # Generate Answer
    # -------------------------
    with st.spinner("Generating answer..."):
        answer = generate_answer(
            question=question,
            vectorstore=vectorstore,
            web_context=web_context
        )

    st.subheader("Answer")
    st.write(answer)

# -------------------------
# Footer
# -------------------------
st.divider()
st.caption("Built with Streamlit • LangChain • FAISS • Groq • Tavily")
