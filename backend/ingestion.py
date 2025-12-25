from langchain.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

def load_documents(folder="documents"):
    docs = []
    for file in Path(folder).iterdir():
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
        elif file.suffix == ".txt":
            loader = TextLoader(str(file))
            docs.extend(loader.load())
    return docs
