import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def ingest_uploaded_files(uploaded_files):
    documents = []

    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        else:
            continue

        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["file_name"] = file.name
            doc.metadata["source"] = f"{file.name} (page {doc.metadata.get('page', 'N/A')})"

        documents.extend(loaded_docs)

    return documents
