import streamlit as st
from backend.ingestion import load_documents
from backend.chunking import chunk_documents
from backend.vector_store import create_vector_store
from backend.web_search import web_search
from backend.router import route_query
from backend.rag import generate_answer

st.set_page_config(page_title="AI Advocate RAG Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📘 About")
    st.write("""
    This AI Advocate RAG Chatbot can:
    - Answer questions from your documents
    - Search the web using Tavily
    - Combine both sources intelligently
    """)
    web_toggle = st.toggle("🌐 Enable Web Search")

    st.divider()
    uploaded_files = st.file_uploader("📄 Upload Documents", accept_multiple_files=True)

# Main UI
st.title("AI Advocate RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Ask a question about your documents")

if st.button("Ask"):
    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = create_vector_store(chunks)

    route = route_query(question, web_toggle)

    context = ""
    if route in ["doc", "hybrid"]:
        docs = vectorstore.similarity_search(question, k=4)
        context += "\n".join(d.page_content for d in docs)

    if route in ["web", "hybrid"]:
        results = web_search(question)
        context += "\n".join(r["content"] for r in results["results"])

    answer = generate_answer(context, question)
    st.session_state.chat.append((question, answer))

# Chat history
for q, a in st.session_state.chat:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")
    if st.toggle("✏️ Edit question"):
        st.text_input("Edit your question", value=q)
