import streamlit as st
from backend.ingestion import load_uploaded_documents
from backend.vector_store import create_vector_store
from backend.rag import generate_answer
from backend.web_search import tavily_search

st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    layout="wide"
)

# -----------------------
# Session State Init
# -----------------------
if "documents" not in st.session_state:
    st.session_state.documents = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("📘 About")
    st.markdown("""
    **Multi-Document RAG Chatbot can:**
    - Answer questions from uploaded documents
    - Search the web using Tavily
    - Combine document + web context intelligently
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

    if uploaded_files:
        with st.spinner("Processing documents..."):
            docs = load_uploaded_documents(uploaded_files)
            st.session_state.documents = docs
            st.session_state.vectorstore = create_vector_store(docs)

        st.success(f"{len(uploaded_files)} document(s) indexed successfully")

# -----------------------
# Main UI
# -----------------------
st.title("📄 Multi-Document RAG Chatbot")

sample_questions = [
    "Summarize the uploaded documents",
    "What is the main topic discussed?",
    "Explain this document in simple terms",
    "List key points from the document"
]

st.markdown("### 💡 Sample Questions")
cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q):
        st.session_state.question = q

question = st.text_input(
    "Ask a question about your documents",
    value=st.session_state.get("question", "")
)

# -----------------------
# Ask Button
# -----------------------
if st.button("Ask"):

    if not st.session_state.vectorstore:
        st.error("Please upload documents first.")
        st.stop()

    web_context = ""
    if enable_web:
        with st.spinner("Searching the web..."):
            web_context = tavily_search(question)

    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(
            question=question,
            vectorstore=st.session_state.vectorstore,
            web_context=web_context,
            hybrid=enable_hybrid
        )

        st.session_state.answer = answer
        st.session_state.sources = sources

# -----------------------
# Output
# -----------------------
if st.session_state.answer:
    st.divider()
    st.subheader("✅ Answer")
    st.write(st.session_state.answer)

    st.subheader("📚 Sources")
    if st.session_state.sources:
        for src in st.session_state.sources:
            st.write(f"• {src}")
    else:
        st.info("No sources available")
