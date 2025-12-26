from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(folder="documents"):
    docs = []

    for file in Path(folder).iterdir():
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())

        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file))
            docs.extend(loader.load())

    return docs
