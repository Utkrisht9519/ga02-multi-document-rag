from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path
import tempfile
import os

def ingest_uploaded_files(uploaded_files):
    """
    Takes Streamlit uploaded files and converts them into LangChain Documents
    """
    documents = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())

            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                documents.extend(loader.load())

        finally:
            os.remove(tmp_path)

    return documents
