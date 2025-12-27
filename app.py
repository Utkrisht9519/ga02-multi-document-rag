import streamlit as st
from backend.ingestion import ingest_uploaded_files
from backend.chunking import chunk_documents
from backend.vector_store import create_vector_store
from backend.web_search import tavily_search
from backend.rag import generate_answer

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    layout="wide"
)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("## 📘 About")
    st.markdown("""
    **Multi-Document RAG Chatbot can:**
    - Answer questions from uploaded documents
    - Search the web using Tavily
    - Combine document + web context intelligently
    """)

    enable_web = st.toggle("🌐 Enable Web Search", value=False)
    enable_hybrid = st.toggle("🔀 Enable Hybrid Search", value=False)

    st.divider()

    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

# -------------------------
# Main UI
# -------------------------
st.title("Multi-Document RAG Chatbot")

st.markdown("### 💡 Sample Questions")
samples = [
    "Summarize the uploaded documents",
    "What is the main topic discussed?",
    "Explain this document in simple terms",
    "List key points from the document"
]

cols = st.columns(len(samples))
for i, q in enumerate(samples):
    if cols[i].button(q):
        st.session_state["question"] = q

question = st.text_input(
    "Ask a question about your documents",
    value=st.session_state.get("question", "")
)

ask_clicked = st.button("Ask")

# -------------------------
# RAG Flow
# -------------------------
if ask_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing documents..."):
            docs = ingest_uploaded_files(uploaded_files)
            chunks = chunk_documents(docs)
            vectorstore = create_vector_store(chunks)

        web_context = None
        if enable_web or enable_hybrid:
            web_context = tavily_search(question)

        with st.spinner("Generating answer..."):
            answer, sources = generate_answer(
                question=question,
                vectorstore=vectorstore,
                web_context=web_context,
                hybrid=enable_hybrid
            )

        # -------------------------
        # Answer Section (ONLY place sources appear)
        # -------------------------
        st.markdown("## ✅ Answer")
        st.write(answer)

        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                st.markdown(f"- {src}")

# -------------------------
# Footer
# -------------------------
st.divider()
st.caption("Built with Streamlit · LangChain · FAISS · Groq · Tavily")
