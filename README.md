🧠 GA02 – Multi-Document RAG Search Engine

with Real-Time Web & Hybrid Search

Name: Utkrisht Agrawal

Email: www.rkayush412@gmail.com

A production-ready Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, LangChain, FAISS, Groq, and Tavily.
This application allows users to upload multiple documents, ask questions, and receive grounded answers with citations, optionally enhanced by real-time web search.

🚀 Key Features

📄 Multi-Document Upload (PDF / TXT)

🔍 Semantic Search using FAISS

🌐 Optional Real-Time Web Search (Tavily)

🔀 Hybrid Search (Documents + Web)

📚 Source-Aware Answers with Citations

⚡ Fast LLM Inference via Groq

💾 Session-safe (no rerun reset issues)

☁️ Streamlit Cloud Deployment Ready

🧩 Architecture Overview

User Query
   ↓
Vector Search (FAISS)
   ↓
[ Optional Web Search (Tavily) ]
   ↓
Context Fusion
   ↓
LLM (Groq – LLaMA 3.1)
   ↓
Answer + Sources

📂 Project Structure

ga02-multi-document-rag/
│
├── app.py                     # Streamlit UI & app logic
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python runtime version
├── README.md                  # Project documentation
│
├── backend/
│   ├── ingestion.py           # File loading & preprocessing
│   ├── vector_store.py        # FAISS vector store creation
│   ├── rag.py                 # RAG pipeline & LLM logic
│   └── web_search.py          # Tavily web search
│
└── .streamlit/
    └── secrets.toml           # API keys (NOT committed)

🛠️ Tech Stack

Component	Technology

UI	Streamlit

Vector DB	FAISS

Embeddings	HuggingFace (MiniLM)

LLM	Groq (LLaMA 3.1)

Web Search	Tavily

Orchestration	LangChain

📦 Installation (Local Setup)

1️⃣ Clone Repository

git clone https://github.com/Utkrisht9519/ga02-multi-document-rag.git

cd ga02-multi-document-rag

2️⃣ Create Virtual Environment

python -m venv venv

venv\Scripts\activate           # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure API Keys

Create file:

.streamlit/secrets.toml


Add:

GROQ_API_KEY = "your_groq_api_key"

TAVILY_API_KEY = "your_tavily_api_key"


⚠️ Never commit secrets.toml to GitHub

5️⃣ Run the App

streamlit run app.py


Open browser at:

http://localhost:8501

☁️ Deployment (Streamlit Cloud)

Push code to GitHub

Open https://streamlit.io/cloud

Create new app

Select:

Repository

Branch: main

File: app.py

Add secrets in Streamlit dashboard:

GROQ_API_KEY

TAVILY_API_KEY


Deploy 🚀

🎯 How to Use

Upload one or more PDF / TXT files

Choose search mode:

Document Search

Web Search

Hybrid Search

Ask questions or use sample prompts

Get:

Natural language answers

Source citations

💡 Sample Questions

Summarize the uploaded documents

What is the main topic discussed?

Explain this document in simple terms

List key points from the document

🧪 Hybrid Search Explained

Mode	                  Behavior

Document     Only	Answers strictly from uploaded files

Web Search	 Uses real-time web data

Hybrid       combines documents + web context intelligently

🧠 Why This Implementation is Stable

✔ Prevents Streamlit rerun resets

✔ CPU-safe embeddings (no Torch crashes)

✔ Session-persistent vector store

✔ Proper source extraction

✔ Python 3.13 compatible

✔ Streamlit Cloud compliant

🐞 Troubleshooting

❌ App resets after upload

✅ Fixed using st.session_state

❌ Torch/embedding crash

✅ Forced CPU embeddings

❌ Sources show as letters

✅ Correct metadata extraction

❌ FAISS / Python mismatch

✅ Compatible FAISS version pinned

📌 Limitations

FAISS index is session-based (not persistent)

No chat memory (yet)

No OCR support (PDF must be text-based)

🔮 Future Enhancements

🔗 Persistent vector DB (Chroma / Qdrant)

🧠 Conversational memory

📑 Page-level citations

📊 Token usage tracking

🔍 Reranker (BM25 + Dense)

👤 Author

Utkrisht Agrawal

🎓 MSc Statistics | AI & Data Analytics

🔗 GitHub: https://github.com/Utkrisht9519
