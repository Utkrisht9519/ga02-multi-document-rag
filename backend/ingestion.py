from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import tempfile
import os

def load_uploaded_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        suffix = file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        else:
            loader = TextLoader(tmp_path)
            docs = loader.load()

        for d in docs:
            d.metadata["source"] = file.name

        documents.extend(docs)
        os.remove(tmp_path)

    return documents
