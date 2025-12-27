from pathlib import Path
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document


def ingest_uploaded_files(uploaded_files):
    """
    Ingest uploaded PDF/TXT files while preserving original filenames.
    """
    all_docs = []
    uploaded_filenames = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        uploaded_filenames.append(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if suffix.lower() == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs = loader.load()

        # Attach original filename to metadata
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    return all_docs, uploaded_filenames
