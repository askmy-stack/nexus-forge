import re


def _word_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long paragraph into fixed-size word windows."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def semantic_chunk(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[str]:
    """Split text on paragraph boundaries with word-level fallback for long paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        stripped = text.strip()
        paragraphs = [stripped] if stripped else [""]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph.split()) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_word_chunks(paragraph, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate.split()) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    chunks = [chunk for chunk in chunks if chunk]
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
