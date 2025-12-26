import streamlit as st
from backend.chunking import chunk_documents
from backend.vector_store import create_vector_store
from backend.rag import generate_answer
from backend.web_search import tavily_search

from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Session State Initialization
# -----------------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📘 About")
    st.markdown(
        """
        **Multi-Document RAG Chatbot can:**
        - Answer questions from uploaded documents
        - Search the web using Tavily
        - Combine document + web context intelligently
        """
    )

    enable_web_search = st.toggle("🌐 Enable Web Search", value=False)

    enable_hybrid_search = False
    if enable_web_search:
        enable_hybrid_search = st.toggle(
            "🔀 Enable Hybrid Search (Docs + Web)",
            value=False
        )

    st.divider()

    st.markdown("## 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )


# -----------------------------
# Title
# -----------------------------
st.title("📄 Multi-Document RAG Chatbot")


# -----------------------------
# Sample Questions
# -----------------------------
st.markdown("### 💡 Sample Questions")

sample_cols = st.columns(4)
sample_questions = [
    "Summarize the uploaded documents",
    "What is the main topic discussed?",
    "Explain this document in simple terms",
    "List key points from the document"
]

for col, q in zip(sample_cols, sample_questions):
    if col.button(q):
        st.session_state.sample_question = q

question = st.text_input(
    "Ask a question about your documents",
    value=st.session_state.get("sample_question", "")
)


# -----------------------------
# Document Processing
# -----------------------------
if uploaded_files:
    if uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files

        docs = []
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    loader = PyPDFLoader(tmp.name)
                    docs.extend(loader.load())

            elif file.name.endswith(".txt"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(file.read())
                    loader = TextLoader(tmp.name)
                    docs.extend(loader.load())

        with st.spinner("Processing documents..."):
            st.session_state.documents = docs
            st.session_state.chunks = chunk_documents(docs)
            st.session_state.vectorstore = create_vector_store(
                st.session_state.chunks
            )

        st.success("✅ Documents uploaded and indexed successfully!")


# -----------------------------
# Question Answering
# -----------------------------
if st.button("Ask"):
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload documents first.")
        st.stop()

    web_context = None
    if enable_web_search:
        with st.spinner("Searching the web..."):
            web_context = tavily_search(question)

    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(
            question=question,
            vectorstore=st.session_state.vectorstore,
            web_context=web_context,
            hybrid=enable_hybrid_search
        )

    st.divider()
    st.markdown("## ✅ Answer")
    st.write(answer)

    if sources:
        st.markdown("## 📚 Sources")
        for src in sources:
            st.markdown(f"- {src}")


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Built with Streamlit · LangChain · FAISS · Groq · Tavily"
)
