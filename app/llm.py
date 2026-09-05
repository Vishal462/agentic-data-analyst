"""Single source of truth for the chat model.

Kept in one place so the model can be swapped (or pointed at a hosted model)
without touching the planner and SQL agent independently.
"""
import os
from langchain_ollama import ChatOllama

MODEL = os.getenv("DATA_ANALYST_MODEL", "qwen3:8b")
# qwen3 emits <think> blocks unless thinking is disabled; that costs latency on
# every call in an agent loop, where most steps need no deliberation.
REASONING = os.getenv("DATA_ANALYST_REASONING", "").lower() in {"1", "true", "on"}

def build_llm(temperature: float = 0) -> ChatOllama:
    return ChatOllama(model=MODEL, temperature=temperature, reasoning=REASONING)
