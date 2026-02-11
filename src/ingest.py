"""
Ingest FDA drug review PDFs into a ChromaDB vector store.

Usage:
    python src/ingest.py              # ingest all PDFs in data/
    python src/ingest.py --reset      # clear existing DB and re-ingest
"""

import os
import sys
import shutil
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#Directory where all the pdfs are stored
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

#Directory to store Chroma DB with emebddings
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Open source Model, I dont want to waste money on this
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# PDF extraction using PyMuPDF
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, returning a list of page-level documents."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                    },
                }
            )
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_documents(pages: list[dict]) -> list:
    """Split page-level text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append(
                {
                    "text": split,
                    "metadata": {
                        **page["metadata"],
                        "chunk": i,
                    },
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# Build vector store
# ---------------------------------------------------------------------------
def build_vectorstore(reset: bool = False):
    """Read all PDFs from data/, chunk them, and persist to ChromaDB."""

    if reset and os.path.exists(CHROMA_DIR):
        print("Resetting existing vector store...")
        shutil.rmtree(CHROMA_DIR)

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/. Add FDA review PDFs and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {DATA_DIR}/")

    # Extract & chunk
    all_chunks = []
    for pdf in pdf_files:
        path = os.path.join(DATA_DIR, pdf)
        print(f"  Processing: {pdf}")
        pages = extract_text_from_pdf(path)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)
        print(f"    → {len(pages)} pages → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Embed & store
    print(f"Embedding with {EMBEDDING_MODEL} (first run downloads the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector store saved to {CHROMA_DIR}/ ({vectorstore._collection.count()} vectors)")
    return vectorstore


# ---------------------------------------------------------------------------
# Load existing vector store
# ---------------------------------------------------------------------------
def load_vectorstore():
    """Load a previously built ChromaDB vector store."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DIR}/. Run `python src/ingest.py` first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    build_vectorstore(reset=reset)
