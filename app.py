import streamlit as st

from backend.ingestion import ingest_uploaded_files
from backend.vector_store import create_vector_store
from backend.web_search import tavily_search
from backend.rag import generate_answer

st.set_page_config(page_title="Multi-Document RAG Chatbot", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("About")

st.sidebar.markdown("""
Multi-Document RAG Chatbot can:
- Answer questions from uploaded documents
- Search the web using Tavily
- Combine document + web context intelligently
""")

web_enabled = st.sidebar.toggle("Enable Web Search", value=False)
hybrid_enabled = st.sidebar.toggle("Enable Hybrid Search", value=False)

st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------- State ----------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if uploaded_files:
    st.session_state.documents = ingest_uploaded_files(uploaded_files)

# ---------------- Main UI ----------------
st.title("Multi-Document RAG Chatbot")

question = st.text_input("Ask a question about your documents")

if st.button("Ask") and question:

    mode = "doc"
    web_context = None

    if hybrid_enabled:
        mode = "hybrid"
    elif web_enabled:
        mode = "web"

    vectorstore = None
    docs_for_rag = []

    if mode in ["doc", "hybrid"] and st.session_state.documents:
        vectorstore = create_vector_store(st.session_state.documents)
        docs_for_rag = vectorstore.similarity_search(question, k=5)

    if mode in ["web", "hybrid"]:
        web_context = tavily_search(question)

    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(
            question=question,
            documents=docs_for_rag,
            web_context=web_context,
            mode=mode
        )

    # ---------------- Answer ----------------
    st.markdown("## ✅ Answer")
    st.write(answer)

    # ---------------- Sources (RULED) ----------------
    st.markdown("### 📚 Sources")

    if mode in ["doc", "hybrid"]:
        if sources["doc_citations"]:
            st.markdown("**Document Citations:**")
            for c in sources["doc_citations"]:
                st.markdown(f"- {c}")

        if not web_enabled:
            st.markdown("**Uploaded Files:**")
            for f in sources["doc_files"]:
                st.markdown(f"- {f}")

    if mode in ["web", "hybrid"]:
        st.markdown("**Web Sources:**")
        for w in sources["web_sources"]:
            st.markdown(f"- {w}")
