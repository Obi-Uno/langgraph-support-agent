"""
Small local RAG layer over the FAQ/policy docs in app/data/faq.

Deliberately implemented with scikit-learn TF-IDF + cosine similarity instead
of a vector DB (Chroma/Pinecone/pgvector). Two reasons:
  1. Zero external dependency at runtime -- no downloading an embedding model
     from Hugging Face on first request, no hosted vector DB that could be
     region-blocked or rate-limited (see: Supabase's India block in Feb 2026).
  2. For a knowledge base this small (a handful of FAQ docs), TF-IDF retrieval
     is plenty accurate and keeps the app deployable on a free-tier instance
     with a fast cold start.

Swapping this for pgvector/Chroma/Pinecone is a ten-line change (retrieve() is
the only function callers depend on) -- the agent graph and everything else
stays identical. At that scale you would also want multi-query retrieval, since
TF-IDF averages a multi-topic question into a single ranking.
"""
import os
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_DIR = os.path.join(os.path.dirname(__file__), "data", "faq")

_vectorizer = None
_doc_matrix = None
_chunks = []


def _chunk(text, chunk_size=400):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) > chunk_size and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += p + "\n\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _build_index():
    global _vectorizer, _doc_matrix, _chunks
    _chunks = []
    for path in sorted(glob.glob(os.path.join(FAQ_DIR, "*.md"))):
        source = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for chunk in _chunk(content):
            _chunks.append({"text": chunk, "source": source})

    if not _chunks:
        _vectorizer = None
        _doc_matrix = None
        return

    _vectorizer = TfidfVectorizer(stop_words="english")
    _doc_matrix = _vectorizer.fit_transform([c["text"] for c in _chunks])


def retrieve(query, k=3):
    """Return top-k relevant FAQ chunks for the given query."""
    global _vectorizer, _doc_matrix
    if _vectorizer is None:
        _build_index()
    if not _chunks or _vectorizer is None:
        return []

    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _doc_matrix)[0]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = []
    for i in ranked[:k]:
        if scores[i] <= 0:
            continue
        hits.append({"text": _chunks[i]["text"], "source": _chunks[i]["source"], "score": float(scores[i])})
    return hits


def retrieve_as_context(query, k=3):
    hits = retrieve(query, k)
    if not hits:
        return "No relevant policy documents found."
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
