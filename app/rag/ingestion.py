import hashlib
from pathlib import Path
from llama_index.core import SimpleDirectoryReader #Converts data into document format that llama index can process
from llama_index.core.node_parser import SentenceSplitter #Split long documents into nodes (smaller text chunks)
from app.rag.vector_store import get_index

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}

def ingest_documents(paths: list[str | Path], session_id: str = "default") -> dict:
    """Ingest user-selected files once; duplicate chunks are not re-embedded."""
    files = [Path(path).resolve() for path in paths]
    if not files: raise ValueError("Choose at least one document.")
    invalid = [str(path) for path in files if path.suffix.lower() not in SUPPORTED_SUFFIXES]
    if invalid: raise ValueError(f"Unsupported document type: {', '.join(invalid)}")
    missing = [str(path) for path in files if not path.is_file()]
    if missing: raise FileNotFoundError(f"Document not found: {', '.join(missing)}")
    documents = SimpleDirectoryReader(input_files=[str(path) for path in files]).load_data()
    if not documents or not any(document.text.strip() for document in documents): raise ValueError("The selected document(s) contain no extractable text.")
    for document in documents:
        source = Path(str(document.metadata.get("file_path") or document.metadata.get("file_name") or "uploaded"))
        document.metadata.update({"source_filename": source.name, "document_type": source.suffix.lower().lstrip("."), "session_id": session_id})
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(documents)
    for number, node in enumerate(nodes):
        digest = hashlib.sha256(f"{node.metadata.get('source_filename')}:{node.text}".encode()).hexdigest()
        node.id_ = digest
        node.metadata["chunk_id"] = number
        node.metadata["content_hash"] = digest
    index, collection = get_index(session_id)
    existing = set(collection.get(ids=[node.id_ for node in nodes])["ids"])
    new_nodes = [node for node in nodes if node.id_ not in existing]
    if new_nodes: index.insert_nodes(new_nodes)
    return {"session_id": session_id, "documents": sorted({node.metadata["source_filename"] for node in nodes}), "chunks_indexed": len(new_nodes), "chunks_skipped": len(nodes) - len(new_nodes)}
