from pathlib import Path

from rag.document_loader import load_document
from rag.chunker import create_chunks
from rag.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_PATH = (
    BASE_DIR
    / "documents"
    / "playwright.txt"
)

VECTOR_DB_PATH = (
    BASE_DIR
    / "vector_db"
)


def main():

    print("Loading Playwright document...")

    document = load_document(
        DOCUMENT_PATH
    )

    print("Creating chunks...")

    chunks = create_chunks(
        document
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating Vector DB...")

    vector_store = VectorStore(
        VECTOR_DB_PATH
    )

    vector_store.add_documents(
        chunks
    )

    print("Vector DB created successfully.")


if __name__ == "__main__":
    main()