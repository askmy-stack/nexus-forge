"""RAG-based summarization: retrieve relevant chunks, then summarize."""

from textSummarizer.models.base import BaseSummarizer
from textSummarizer.pipelines.chunking import semantic_chunk
from textSummarizer.pipelines.stuff import stuff_summarize


def _bm25_scores(query: str, chunks: list[str]) -> list[float]:
    from rank_bm25 import BM25Okapi

    tokenized = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return list(bm25.get_scores(query.split()))


def _embedding_scores(query: str, chunks: list[str]) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    return [float(query_embedding @ chunk_emb) for chunk_emb in chunk_embeddings]


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def retrieve_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 5,
    use_embeddings: bool = True,
) -> list[str]:
    """Rank chunks with BM25 and optional dense retrieval."""
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    bm25 = _normalize(_bm25_scores(query, chunks))
    if use_embeddings:
        try:
            dense = _normalize(_embedding_scores(query, chunks))
            combined = [0.5 * b + 0.5 * d for b, d in zip(bm25, dense, strict=True)]
        except ImportError:
            combined = bm25
    else:
        combined = bm25

    ranked = sorted(
        zip(combined, chunks, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    return [chunk for _, chunk in ranked[:top_k]]


def rag_summarize(
    text: str,
    summarizer: BaseSummarizer,
    max_length: int = 128,
    query: str | None = None,
    top_k: int = 5,
    chunk_size: int = 400,
    overlap: int = 50,
) -> str:
    chunks = semantic_chunk(text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) <= 1:
        return stuff_summarize(text, summarizer, max_length=max_length)

    retrieval_query = query or text[:500]
    try:
        selected = retrieve_chunks(retrieval_query, chunks, top_k=top_k)
    except ImportError as exc:
        raise ImportError(
            "RAG strategy requires optional dependencies. Install with: "
            "pip install nexus-forge[rag]"
        ) from exc

    context = "\n\n".join(selected)
    prompt = f"Summarize the following relevant passages:\n\n{context}"
    return stuff_summarize(prompt, summarizer, max_length=max_length)
