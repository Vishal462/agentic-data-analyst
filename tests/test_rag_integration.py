import os

import pytest


pytestmark = pytest.mark.skipif(os.getenv("RUN_RAG_INTEGRATION") != "1", reason="Downloads local embeddings and writes a temporary Chroma store.")


def test_ingestion_and_source_tracking(tmp_path, monkeypatch):
    pytest.importorskip("chromadb"); pytest.importorskip("llama_index")
    from app.rag import vector_store
    monkeypatch.setattr(vector_store, "CHROMA_PATH", tmp_path / "chroma")
    from app.rag.ingestion import ingest_documents
    from app.rag.retriever import retrieve
    document = tmp_path / "metric.md"; document.write_text("Margin means profit divided by revenue.", encoding="utf-8")
    result = ingest_documents([document], session_id="test")
    assert result["chunks_indexed"] > 0
    retrieved = retrieve("What does margin mean?", session_id="test")
    assert retrieved["status"] == "ok" and retrieved["sources"][0]["filename"] == "metric.md"


def test_multiple_document_metadata(tmp_path, monkeypatch):
    pytest.importorskip("chromadb"); pytest.importorskip("llama_index")
    from app.rag import vector_store
    monkeypatch.setattr(vector_store, "CHROMA_PATH", tmp_path / "chroma")
    from app.rag.ingestion import ingest_documents
    from app.rag.retriever import retrieve
    first, second = tmp_path / "one.txt", tmp_path / "two.txt"
    first.write_text("Alpha is a planning metric.", encoding="utf-8"); second.write_text("Beta is an operational metric.", encoding="utf-8")
    ingest_documents([first, second], session_id="test")
    assert {source["filename"] for source in retrieve("What is Beta?", "test")["sources"]} <= {"one.txt", "two.txt"}
