def create_chunks(text: str,chunk_size: int = 5) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    chunks = []
    for i in range(0,len(paragraphs),chunk_size):
        chunk = "\n".join(paragraphs[i:i + chunk_size])
        chunks.append(chunk)
    return chunks