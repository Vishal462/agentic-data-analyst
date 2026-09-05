"""Temporary runtime document-upload interface until a UI is added."""
import argparse
import json
from app.rag.ingestion import ingest_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index user-provided RAG documents.")
    parser.add_argument("paths", nargs="+", help="PDF, TXT, Markdown, or DOCX files")
    parser.add_argument("--session", default="default", help="Document collection/session identifier")
    args = parser.parse_args()
    print(json.dumps(ingest_documents(args.paths, args.session), indent=2))
