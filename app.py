import os
import streamlit as st
from backend.ingestion import load_documents
from backend.chunking import chunk_documents
from backend.vector_store import create_vector_store
from backend.rag import generate_answer
from backend.web_search import tavily_search

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("📘 About")

    st.markdown("""
    **Multi-Document RAG Chatbot** can:

    • Answer questions from uploaded documents  
    • Search the web using Tavily  
    • Combine document + web context intelligently  
    """)

    enable_web = st.toggle("🌐 Enable Web Search", value=False)
    enable_hybrid = st.toggle("🔀 Enable Hybrid Search", value=False)

    st.divider()

    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

# -------------------------
# Ensure documents folder
# -------------------------
DOC_DIR = "documents"
os.makedirs(DOC_DIR, exist_ok=True)

# -------------------------
# Save Uploaded Files
# -------------------------
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DOC_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

    st.success(f"Uploaded {len(uploaded_files)} document(s)")

# -------------------------
# Main UI
# -------------------------
st.title("Multi-Document RAG Chatbot")

# Sample Questions
st.subheader("💡 Sample Questions")

sample_questions = [
    "Summarize the uploaded documents",
    "What is the main topic discussed?",
    "Explain this document in simple terms",
    "List key points from the document",
]

cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q):
        st.session_state["question"] = q

# -------------------------
# Question Input
# -------------------------
question = st.text_input(
    "Ask a question about your documents",
    value=st.session_state.get("question", "")
)

# -------------------------
# Ask Button
# -------------------------
if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    docs = load_documents(DOC_DIR)

    if not docs:
        st.error("No documents found. Please upload files first.")
        st.stop()

    chunks = chunk_documents(docs)
    vectorstore = create_vector_store(chunks)

    web_context = ""
    if enable_web:
        web_context = tavily_search(question)

    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(
            question=question,
            vectorstore=vectorstore,
            web_context=web_context,
            hybrid=enable_hybrid
        )

    # -------------------------
    # Answer Display
    # -------------------------
    st.subheader("✅ Answer")
    st.write(answer)

    # -------------------------
    # Citations
    # -------------------------
    if sources:
        st.subheader("📚 Sources")
        for src in sources:
            st.markdown(f"- {src}")

# -------------------------
# Footer
# -------------------------
st.divider()
st.caption("Built with Streamlit • LangChain • FAISS • Groq • Tavily")
