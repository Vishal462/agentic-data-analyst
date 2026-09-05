"""Deterministic evaluation of the agent, recorded in LangSmith.

    python tests/langsmith_eval.py --smoke        first 3 cases, verifies the harness
    python tests/langsmith_eval.py                the full flight dataset
    python tests/langsmith_eval.py --sales        the non-flight cases (registers a sales table)
    python tests/langsmith_eval.py --local        skip LangSmith, print a local report only

No LLM judges: every score is computed from the agent's own tool calls and tool
outputs, so the same run always produces the same score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from app.agent import MAX_STEPS, _ensure_citations, analyst_agent  # noqa: E402
from app.tools import current_session, reset_results  # noqa: E402

CASES_PATH = Path(__file__).parent / "eval_cases.json"
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
# Values a grounded answer may legitimately contain even though no tool emitted them.
FREE_NUMBERS = {0.0, 0.05, 1.0, 100.0}


# --------------------------------------------------------------------------- run

def run_agent(question: str, session_id: str) -> dict:
    """Run the graph and capture the full tool calls and tool outputs.

    stream_events truncates tool results for display; grounding checks need the
    whole payload, so this consumes the message stream directly.
    """
    current_session.set(session_id)
    reset_results()
    tool_calls: list[dict] = []
    tool_outputs: list[dict] = []
    answer, steps, errors = "", 0, []
    started = time.perf_counter()
    for update in analyst_agent.stream({"messages": [HumanMessage(content=question)], "steps": 0},
                                       config={"recursion_limit": MAX_STEPS * 2 + 5},
                                       stream_mode="updates"):
        for delta in update.values():
            steps = max(steps, delta.get("steps", 0))
            for message in delta.get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls or []:
                        tool_calls.append({"name": call["name"], "args": call["args"]})
                    if not message.tool_calls and message.content:
                        answer = message.content
                elif isinstance(message, ToolMessage):
                    try:
                        payload = json.loads(message.content)
                    except (json.JSONDecodeError, TypeError):
                        payload = {"_raw": message.content}
                    tool_outputs.append({"name": message.name, "payload": payload})
                    if isinstance(payload, dict) and "error" in payload:
                        errors.append(f"{message.name}: {str(payload['error'])[:120]}")
    # The graph is consumed directly here, so apply the same delivery-time citation
    # guarantee stream_events applies; otherwise this grades an answer no user sees.
    retrieved = [name for output in tool_outputs
                 if isinstance(output["payload"], dict)
                 for name in (output["payload"].get("cite_these_filenames") or [])]
    answer = _ensure_citations(answer, list(dict.fromkeys(retrieved)))
    return {"answer": answer, "tool_calls": tool_calls, "tool_outputs": tool_outputs,
            "tools": [call["name"] for call in tool_calls], "steps": steps,
            "errors": errors, "latency_s": round(time.perf_counter() - started, 1)}


# ------------------------------------------------------------------- extraction

def numbers_in(text: str) -> list[float]:
    """Numbers a reader would take as claims. Ordinals in '1. Foo' are list markers."""
    cleaned = text.translate(SUPERSCRIPTS)
    cleaned = re.sub(r"(?m)^\s*\d+[.)]\s", " ", cleaned)          # numbered list markers
    cleaned = cleaned.replace("\\times", "x").replace("\\cdot", "x").replace("$", "")
    # 4.47 x 10^-47, 4.47 x 10^{-47} and 4.47 × 10⁻⁴⁷ are all one number, not two.
    cleaned = re.sub(r"(\d)\s*[x×*]\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?", r"\1e\2", cleaned)
    found = []
    for token in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", cleaned):
        try:
            found.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return found


def numbers_from(value, into: list[float]) -> list[float]:
    """Every numeric value a tool actually returned, at any nesting depth."""
    if isinstance(value, bool):
        return into
    if isinstance(value, (int, float)):
        into.append(float(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            numbers_from(key, into)      # "25%", "50%" are labels an answer may quote
            numbers_from(item, into)
    elif isinstance(value, list):
        for item in value:
            numbers_from(item, into)
    elif isinstance(value, str):
        for token in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", value):
            try:
                into.append(float(token))
            except ValueError:
                continue
    return into


def grounded(claim: float, sources: list[float]) -> bool:
    """A claim is grounded if a tool produced it, or a rounding of it."""
    for source in sources:
        if claim == source:
            return True
        if source and math.isclose(claim, source, rel_tol=1e-6, abs_tol=1e-9):
            return True
        for places in range(0, 5):                      # 18.90 stands for 18.9038
            if round(source, places) == claim:
                return True
        if source and claim and math.isclose(math.log10(abs(claim)) if claim else 0,
                                             math.log10(abs(source)) if source else 0,
                                             rel_tol=1e-3) and abs(claim) < 1e-6:
            return True                                 # p-values quoted to 2 s.f.
    return False


# -------------------------------------------------------------------- evaluators

def correct_tool(case: dict, result: dict) -> dict | None:
    expected = case.get("expected_tool")
    if not expected:
        return None
    used = result["tools"]
    ok = expected in used
    alternatives = case.get("expected_tools_any")
    detail = f"expected {expected}, used {used or 'none'}"
    if alternatives:
        ok = ok and any(name in used for name in alternatives)
        detail += f"; also needed one of {alternatives}"
    return {"key": "correct_tool", "score": int(ok), "comment": detail}


def correct_columns(case: dict, result: dict) -> dict | None:
    """The tool must be called on the columns the question named.

    Without this, an agent that groups by the wrong column still scores green on
    every other check: the tool is right, the numbers are grounded, the prose reads
    well - and the answer is about something the user never asked for.
    """
    wanted = {"group_by": case.get("expected_group_by"), "metric": case.get("expected_metric")}
    if not any(wanted.values()):
        return None
    called = json.dumps(result["tool_calls"], default=str).lower()
    missing = [f"{role} '{name}'" for role, name in wanted.items()
               if name and name.lower() not in called]
    used = [call["args"] for call in result["tool_calls"]]
    return {"key": "correct_columns", "score": int(not missing),
            "comment": ("; ".join(missing) + f" not in any tool call; called {json.dumps(used, default=str)[:180]}")
            if missing else f"used {', '.join(f'{k}={v}' for k, v in wanted.items() if v)}"}


def no_fabricated_numbers(case: dict, result: dict) -> dict:
    """Every number in the answer must trace to a value some tool returned."""
    sources: list[float] = []
    for output in result["tool_outputs"]:
        numbers_from(output["payload"], sources)
    sources.extend(FREE_NUMBERS)
    claims = numbers_in(result["answer"])
    ungrounded = [claim for claim in claims if not grounded(claim, sources)]
    placeholders = re.findall(r"\b(?:Airline|Category|Group|Company)\s+[A-Z]\b|\b[XYZ]\s+minutes\b",
                              result["answer"])
    ok = not ungrounded and not placeholders
    comment = f"{len(claims)} numeric claim(s), {len(ungrounded)} ungrounded"
    if ungrounded:
        comment += f": {ungrounded[:6]}"
    if placeholders:
        comment += f"; placeholders {placeholders[:3]}"
    return {"key": "no_fabricated_numbers", "score": int(ok), "comment": comment}


def groups_reported(case: dict, result: dict) -> dict | None:
    expected = case.get("expected_group_count")
    if expected is None:
        return None
    listed = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S", result["answer"]))
    return {"key": "groups_reported", "score": int(listed == expected),
            "comment": f"expected {expected} listed group(s), found {listed}"}


def stat_correct(case: dict, result: dict) -> dict | None:
    expected_p = case.get("expected_p_value")
    expected_significant = case.get("expected_significant")
    if expected_p is None and expected_significant is None:
        return None
    tests = [output["payload"]["statistical_test"] for output in result["tool_outputs"]
             if isinstance(output["payload"], dict) and isinstance(output["payload"].get("statistical_test"), dict)]
    if not tests:
        return {"key": "stat_correct", "score": 0, "comment": "no statistical_test in any tool output"}
    test = tests[-1]
    problems = []
    if expected_p is not None:
        actual = test.get("p_value")
        if actual is None or not math.isclose(actual, expected_p, rel_tol=1e-3, abs_tol=1e-60):
            problems.append(f"p_value {actual!r} != {expected_p!r}")
    if expected_significant is not None and test.get("significant") != expected_significant:
        problems.append(f"significant {test.get('significant')} != {expected_significant}")
    if "p-value" not in result["answer"].lower() and "p value" not in result["answer"].lower():
        problems.append("answer does not report a p-value")
    return {"key": "stat_correct", "score": int(not problems),
            "comment": "; ".join(problems) or f"p={test.get('p_value'):.3e}, significant={test.get('significant')}"}


def cited_source(case: dict, result: dict) -> dict | None:
    expected = case.get("expected_source")
    if not expected:
        return None
    in_answer = expected.lower() in result["answer"].lower()
    in_tools = any(expected.lower() in json.dumps(output["payload"], default=str).lower()
                   for output in result["tool_outputs"])
    return {"key": "cited_source", "score": int(in_answer and in_tools),
            "comment": f"'{expected}' in answer={in_answer}, returned by a tool={in_tools}"}


def expected_content(case: dict, result: dict) -> dict | None:
    """Required substrings, and the unanswerable cases that must not assert numbers."""
    required = case.get("expected_text") or []
    must_not = case.get("must_not_contain_numbers")
    expected_numbers = case.get("expected_numbers") or []
    if not required and not must_not and not expected_numbers:
        return None
    answer = result["answer"].lower()
    missing = [item for item in required if item.lower() not in answer]
    problems = [f"missing text {missing}"] if missing else []
    sources: list[float] = []
    for output in result["tool_outputs"]:
        numbers_from(output["payload"], sources)
    # The answer legitimately rounds the reference (18.90 for 18.9038), so the
    # answer's number is the claim and the reference is the source, not the reverse.
    stated = numbers_in(result["answer"])
    for wanted in expected_numbers:
        if not any(grounded(claim, [float(wanted)]) for claim in stated):
            problems.append(f"answer does not state {wanted} (found {stated[:8]})")
    if must_not:
        claims = [value for value in numbers_in(result["answer"]) if abs(value) > 1]
        if claims:
            problems.append(f"asserted numbers for an unanswerable question: {claims[:5]}")
    return {"key": "expected_content", "score": int(not problems), "comment": "; ".join(problems) or "ok"}


EVALUATORS = [correct_tool, correct_columns, no_fabricated_numbers, groups_reported,
              stat_correct, cited_source, expected_content]


def score_case(case: dict, result: dict) -> list[dict]:
    scores = []
    for evaluator in EVALUATORS:
        outcome = evaluator(case, result)
        if outcome is not None:
            scores.append(outcome)
    return scores


# -------------------------------------------------------------------------- main

def load_cases(which: str) -> tuple[dict, list[dict]]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = data["sales_cases"] if which == "sales" else data["cases"]
    return data, cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic agent evaluation.")
    parser.add_argument("--smoke", action="store_true", help="Only the first 3 cases.")
    parser.add_argument("--sales", action="store_true", help="Run the non-flight cases.")
    parser.add_argument("--local", action="store_true", help="Do not upload results to LangSmith.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--case", default="", help="Run one case by id.")
    args = parser.parse_args()

    data, cases = load_cases("sales" if args.sales else "flights")
    if args.case:
        cases = [case for case in cases if case["id"] == args.case] or cases
    elif args.smoke:
        cases = cases[:3]
    elif args.limit:
        cases = cases[: args.limit]

    if args.sales:
        from app.db import register_dataset
        sales = Path(__file__).parent / "fixtures" / "sales.csv"
        print(f"Registering {sales.name} as the active dataset...")
        print(" ", register_dataset(sales))

    session_id = data.get("session_id", "default")
    print(f"\n{len(cases)} case(s) | session '{session_id}' | LangSmith upload: {not args.local}\n")

    records, totals = [], {}
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question'][:80]}")
        result = run_agent(case["question"], session_id)
        scores = score_case(case, result)
        for score in scores:
            bucket = totals.setdefault(score["key"], [0, 0])
            bucket[0] += score["score"]
            bucket[1] += 1
            mark = "PASS" if score["score"] else "FAIL"
            print(f"      {mark}  {score['key']:22} {score['comment'][:110]}")
        print(f"      {result['latency_s']}s | {result['steps']} model steps | "
              f"{len(result['tool_calls'])} tool call(s) | {len(result['errors'])} tool error(s)")
        if any(not score["score"] for score in scores):
            for call in result["tool_calls"]:
                print(f"      called {call['name']}({json.dumps(call['args'], default=str)[:120]})")
            print(f"      answer: {result['answer'][:600]!r}")
        records.append({"case": case, "result": result, "scores": scores})

    print(f"\n{'=' * 78}\nDeterministic scores")
    for key, (passed, total) in sorted(totals.items()):
        print(f"  {key:24} {passed}/{total}")
    overall_pass = sum(passed for passed, _ in totals.values())
    overall_total = sum(total for _, total in totals.values())
    print(f"  {'TOTAL':24} {overall_pass}/{overall_total}")
    latencies = [record["result"]["latency_s"] for record in records]
    print(f"\nRun metadata: median latency {sorted(latencies)[len(latencies) // 2]}s | "
          f"total {sum(latencies):.0f}s | "
          f"{sum(len(r['result']['tool_calls']) for r in records)} tool calls | "
          f"{sum(r['result']['steps'] for r in records)} LLM turns | "
          f"{sum(len(r['result']['errors']) for r in records)} tool errors")

    failures = [(r["case"]["id"], s["key"], s["comment"])
                for r in records for s in r["scores"] if not s["score"]]
    if failures:
        print(f"\n{len(failures)} failing check(s):")
        for case_id, key, comment in failures:
            print(f"  {case_id:28} {key:22} {comment[:120]}")

    if not args.local:
        upload(data, records)
    return 0 if not failures else 1


def upload(data: dict, records: list[dict]) -> None:
    """Push examples and scores to LangSmith so runs are comparable over time."""
    try:
        from langsmith import Client
        client = Client()
        name = data["dataset"]
        dataset = next((d for d in client.list_datasets(dataset_name=name)), None)
        if dataset is None:
            dataset = client.create_dataset(dataset_name=name, description=data["description"])
            print(f"\nCreated LangSmith dataset '{name}'.")
        existing = {example.inputs.get("id") for example in client.list_examples(dataset_id=dataset.id)}
        added = 0
        for record in records:
            case = record["case"]
            if case["id"] in existing:
                continue
            client.create_example(
                dataset_id=dataset.id,
                inputs={"id": case["id"], "question": case["question"], "category": case.get("category", "")},
                outputs={key: value for key, value in case.items()
                         if key.startswith("expected") or key == "must_not_contain_numbers"})
            added += 1
        print(f"LangSmith dataset '{name}': {added} example(s) added, "
              f"{len(existing) + added} total. Scores are printed above and each run is "
              f"traced in project '{os.getenv('LANGSMITH_PROJECT', 'default')}'.")
    except Exception as exc:
        print(f"\nLangSmith upload skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
