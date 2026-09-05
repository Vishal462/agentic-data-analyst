from pathlib import Path
import re
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def _collection_name(session_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]", "_", session_id.lower())[:48]
    return f"documents_{safe or 'default'}"


def _existing_collection(session_id: str):
    """Open a collection without creating it.

    get_or_create_collection would make inspection a write: reading an unknown
    session once was enough to add an empty session to the picker.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        return client.get_collection(_collection_name(session_id))
    except Exception:
        return None


def collection_count(session_id: str) -> int:
    """Chunks indexed for a session, without loading the embedding model."""
    collection = _existing_collection(session_id)
    return collection.count() if collection else 0


def list_sessions() -> dict[str, int]:
    """Every indexed session and its chunk count, so past sessions can be found again."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    prefix = "documents_"
    sessions = {collection.name[len(prefix):]: collection.count()
                for collection in client.list_collections() if collection.name.startswith(prefix)}
    return dict(sorted(sessions.items()))


def delete_session(session_id: str) -> None:
    """Drop one session's collection. Only the index is removed; source files are untouched."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client.delete_collection(_collection_name(session_id))


def session_chunks(session_id: str) -> dict[str, list[dict]]:
    """Indexed chunks grouped by source document, so their content can be inspected.

    What the retriever can actually see is what was chunked, not what the file
    contains - being able to read it is how you tell a bad answer from a bad index.
    """
    collection = _existing_collection(session_id)
    if collection is None:
        return {}
    stored = collection.get(include=["documents", "metadatas"])
    grouped: dict[str, list[dict]] = {}
    for text, metadata in zip(stored.get("documents") or [], stored.get("metadatas") or []):
        entry = metadata or {}
        grouped.setdefault(entry.get("source_filename", "unknown"), []).append(
            {"chunk_id": entry.get("chunk_id"), "type": entry.get("document_type", ""), "text": text or ""})
    for chunks in grouped.values():
        chunks.sort(key=lambda chunk: (chunk["chunk_id"] is None, chunk["chunk_id"]))
    return dict(sorted(grouped.items()))


def indexed_documents(session_id: str) -> dict[str, int]:
    """Filename -> chunk count for one session; session_chunks adds the text itself."""
    return {name: len(chunks) for name, chunks in session_chunks(session_id).items()}


def get_index(session_id: str) -> tuple[VectorStoreIndex, object]:
    """Open the persistent collection for one caller-provided session id."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(_collection_name(session_id))
    vector_store = ChromaVectorStore(chroma_collection=collection)
    context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=context, embed_model=embed_model), collection
