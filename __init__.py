"""Application package.

Environment is loaded here so that every entry point (graph, CLI, tests)
gets LangSmith tracing configuration before LangChain/LangGraph are imported.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
