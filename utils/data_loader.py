import json

from pathlib import Path

def load_test_data(file_path: Path) -> list:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Test data not found: {file_path}")
    with open(file_path,"r",encoding="utf-8") as file:
        return json.load(file)