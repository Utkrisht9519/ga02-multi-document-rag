from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(folder="documents"):
    docs = []
    path = Path(folder)

    if not path.exists():
        return docs   # ✅ prevent crash

    for file in path.iterdir():
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
        elif file.suffix == ".txt":
            loader = TextLoader(str(file))
            docs.extend(loader.load())

    return docs
