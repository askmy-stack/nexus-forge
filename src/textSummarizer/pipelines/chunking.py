import re


def semantic_chunk(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[str]:
    """Split text on paragraph boundaries with approximate token overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate.split()) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev_words = chunks[i - 1].split()
        prefix = " ".join(prev_words[-overlap:])
        overlapped.append(f"{prefix} {chunk}".strip())

    return overlapped
