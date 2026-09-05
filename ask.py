"""Command-line runner for the analyst.
  python ask.py "question"                  one question through the agent
  python ask.py "q1" "q2" "q3"              several questions in a row
  python ask.py                             interactive; blank line or 'quit' exits
  python ask.py --session mydocs "question" pick the document collection for RAG
  python ask.py --trace "question"          also print every tool call and result
"""
import argparse
import os
import sys
import time
import warnings

# Library startup noise (HF progress bars, model load reports, pydantic v1 notices)
# drowns the actual answer; silence it before anything imports transformers.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_agent(question: str, session: str, trace: bool) -> None:
    """Print progress as it happens; a question can take minutes, so silence looks broken."""
    import json

    from app.agent import stream_events
    start = time.perf_counter()
    for event in stream_events(question, session):
        elapsed = round(time.perf_counter() - start, 1)
        kind = event["type"]
        if kind == "tool_call":
            print(f"  [{elapsed:>6}s] -> {event['name']}({json.dumps(event['args'], default=str)[:120]})")
        elif kind == "tool_result":
            status = "ok" if event["ok"] else "ERROR"
            print(f"  [{elapsed:>6}s] <- {event['name']}: {status}"
                  + (f" - {event['detail'][:160]}" if trace or not event["ok"] else ""))
        elif kind == "trace":
            print(f"  [{elapsed:>6}s] trace: {event['url']}")
        elif kind == "chart":
            print(f"  [{elapsed:>6}s] chart saved: {event['path']}")
        elif kind == "done":
            print(f"\n{event['answer']}\n")
            print(f"[{elapsed}s | {event['steps']} steps | tools: {', '.join(event['tools_used']) or 'none'}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the data analyst a question.")
    parser.add_argument("questions", nargs="*", help="One or more questions. Omit for interactive mode.")
    parser.add_argument("--session", default="default", help="Document collection for RAG (default: default)")
    parser.add_argument("--trace", action="store_true", help="Print each tool call and result (agent only)")
    args = parser.parse_args()

    runner = lambda q: run_agent(q, args.session, args.trace)
    print(f"[agent | session: {args.session}]")

    questions = args.questions
    if questions:
        for question in questions:
            print(f"\n{'=' * 78}\nQ: {question}")
            try:
                runner(question)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
        return

    print("Type a question, or a blank line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question or question.lower() in {"quit", "exit"}:
            return
        try:
            runner(question)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
