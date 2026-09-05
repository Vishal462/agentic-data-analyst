from app.rag.vector_store import get_index

def retrieve(question: str, session_id: str = "default", top_k: int = 4) -> dict:
    """Return traceable context, never an opaque answer string."""
    index, collection = get_index(session_id)
    if collection.count() == 0:
        return {"status": "empty", "context": "", "sources": [], "chunks": []}
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(question)
    chunks = [{"text": item.node.get_content(), "score": float(item.score or 0), "metadata": dict(item.node.metadata)} for item in nodes]
    sources = [{"filename": chunk["metadata"].get("source_filename"), "document_type": chunk["metadata"].get("document_type"), "chunk_id": chunk["metadata"].get("chunk_id")} for chunk in chunks]
    return {"status": "ok", "context": "\n\n".join(chunk["text"] for chunk in chunks), "sources": sources, "chunks": chunks}
