from pathlib import Path

def load_document(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    return file_path.read_text(encoding="utf-8")