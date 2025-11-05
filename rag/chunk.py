from typing import List

def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> List[str]:
    # Simple whitespace chunking
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks
