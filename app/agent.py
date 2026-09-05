"""The analyst: a LangGraph agent <-> tools loop.
Nothing about the path is decided in advance - the model picks the tools, the
order, and when it has enough to answer. The only code-enforced limits are the
read-only database connection, the tool contracts, and a cap on loop iterations.
app.analyst holds the earlier deterministic pipeline. It is kept as a baseline
for comparison and is no longer wired into the application."""
import json
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from app.llm import build_llm
from app.tools import TOOLS, current_session, reset_results

MAX_STEPS = 12 #Can be changed, maximum number of steps for one question

def _system_prompt() -> str:
    """Built once at import; the schema inline removes a describe_table round trip."""
    from app.tools import schema_summary
    return f"""You are a data analyst. You answer questions about a database by using tools.

Schema (use these exact column names, they are case-sensitive):
{schema_summary()}

Choosing a tool:
1. The column names above are complete and exact. Use them directly - do not invent
   lowercase or spelled-out variants. Call describe_table only if you need more example
   values or missing-value counts.
   When two columns have similar names, pick the one whose example value matches what
   the user described: a code, an identifier, a full name and a city are different
   columns, and filtering the wrong one returns zero rows instead of an error.
   If the question names a grouping, metric or filter that has NO matching column in
   the schema above, say so and list the columns that do exist. Never substitute a
   different column for the one asked for: answering about a column the user did not
   ask for, without saying so, is worse than reporting that the data is not there.
2. Any per-group summary is a compare_groups call - both "show/display <metric> by <group>"
   and "which <groups> have the highest/lowest <metric>", with or without "is the difference
   significant". It aggregates in the database and returns the groups AND the significance
   test in one step. Use it instead of run_sql whenever the question summarises by group.
   Pass `limit` ONLY when the user asked for a specific number or a shortlist ("top 10",
   "the 3 worst"). If they just asked to see the metric by group, omit `limit` entirely so
   every group is returned - do not invent a cutoff they did not ask for.
3. Use run_sql for everything else: totals, counts, filters, single values, or rows to chart.
   Quote identifiers with double quotes and select only the columns you need; SELECT * is rejected.
   run_sql returns at most a 3-row preview, so never try to answer a multi-row question from it.
4. correlate, find_outliers, and describe_distribution take a result_id from run_sql.
5. Call make_chart only if the user asked for a chart, plot, or graph.
6. Use search_documents for what a metric means or how it is defined.
7. If a tool returns an error, read it, correct your call, and try again.

Writing the final answer:
- Before answering, check that the columns you actually used are the ones the question
  asked about. If the question asked to group by, filter on, or measure something with no
  column in the schema, reply "There is no <name> column in this data" and list the
  columns that exist. Presenting results for a different column - region totals when
  the user asked about country - answers a question nobody asked and is a wrong answer,
  even though every number in it is real.
- Answer every part of the question. A two-part question needs both parts answered.
- If you ranked groups, list them with their metric values - all of the ones you were asked for.
- If you ran a statistical test, report the test name, the p-value, and what it means.
- Distinguish statistical significance from practical importance: a small effect can be
  statistically significant in a large dataset. Report the effect size when you have one.
- An ANOVA says at least one group differs. It never says which pairs differ.
- Every number must come from a tool result. Never estimate or invent values.
- If a tool returned an error, you MUST fix the call and run it again before answering.
  Never write placeholders like "Airline A" or "X minutes" - if you do not have the real
  numbers, call the tool again until you do.
- If a statistical_test reports a "scope", state that scope rather than implying the test
  covered only the groups you listed.
- Never describe preview rows instead of the analysis you were asked for.
- Cite the source filenames that search_documents returned.
"""

llm_with_tools = build_llm().bind_tools(TOOLS)
tools_by_name = {tool.name: tool for tool in TOOLS}

_prompt_cache: dict[tuple, str] = {}

def system_prompt() -> str:
    """Rebuilt whenever the active dataset changes, so an upload is visible at once."""
    from app.db import schema_signature
    signature = schema_signature()
    if signature not in _prompt_cache:
        _prompt_cache.clear()
        _prompt_cache[signature] = _system_prompt()
    return _prompt_cache[signature]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int

def call_model(state: AgentState) -> AgentState:
    """One turn of reasoning: the model either calls tools or produces the answer."""
    if state.get("steps", 0) >= MAX_STEPS:
        return {"messages": [AIMessage(content=(
            "I stopped after the maximum number of tool steps without reaching a "
            "confident answer. The partial findings are in the steps above."))],
            "steps": state.get("steps", 0)}
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt())] + state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}

def call_tools(state: AgentState) -> AgentState:
    """Execute every tool the model asked for; failures come back as content, not exceptions.
    Returning the error to the model is what replaces the fixed graph's repair node:
    the agent sees what went wrong and decides how to recover."""
    outputs = []
    for call in state["messages"][-1].tool_calls:
        tool = tools_by_name.get(call["name"])
        if tool is None:
            content = json.dumps({"error": f"Unknown tool {call['name']}. Available: {sorted(tools_by_name)}"})
        else:
            try:
                content = tool.invoke(call["args"])
            except Exception as exc:  # a crashing tool must not kill the run
                content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        outputs.append(ToolMessage(content=content, name=call["name"], tool_call_id=call["id"]))
    return {"messages": outputs, "steps": state.get("steps", 0)}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

#Making the Agent<->Tools Graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
analyst_agent = graph.compile()


def _payload(message: ToolMessage) -> dict:
    """A tool's JSON reply, or an empty dict if it did not return JSON."""
    try:
        parsed = json.loads(message.content)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _ensure_citations(answer: str, retrieved_files: list[str]) -> str:
    """Guarantee retrieved documents are credited, using only filenames a tool returned.
    The prompt asks for citations and the pure-document answers comply, but on a
    two-part question the model drops them. Nothing is invented here: these are the
    exact filenames search_documents returned, appended only when none appear."""
    if not answer or not retrieved_files:
        return answer
    if any(name.lower() in answer.lower() for name in retrieved_files):
        return answer
    return f"{answer.rstrip()}\n\nSource: {', '.join(retrieved_files)}"


def stream_events(question: str, session_id: str = "default"):
    """Yield the run as it happens, one event per tool call, tool result, and answer.
    LangGraph's stream_mode="updates" emits {node_name: state_delta} each time a
    node finishes, so the agent node's delta carries the tool calls it just chose
    and the tools node's delta carries their results. Translating those deltas into
    flat events keeps LangGraph's shape out of the caller: a UI renders events and
    never touches graph state.
    Event types: tool_call, tool_result, chart, answer, and a final `done` carrying
    the same fields ask() returns."""
    current_session.set(session_id)
    reset_results()
    answer, steps, tools_used, chart_path, trace_url = "", 0, [], None, None
    retrieved_files: list[str] = []

    # Reuses the tracer that LANGSMITH_TRACING already installs (verified: no second
    # tracer is added), purely so the run's LangSmith URL can be surfaced live.
    with tracing_v2_enabled() as tracer:
        for update in analyst_agent.stream({"messages": [HumanMessage(content=question)], "steps": 0},
                                           config={"recursion_limit": MAX_STEPS * 2 + 5},
                                           stream_mode="updates"):
            for delta in update.values():
                steps = max(steps, delta.get("steps", 0))
                for message in delta.get("messages", []):
                    if isinstance(message, ToolMessage):
                        payload = _payload(message)
                        tools_used.append(message.name)
                        yield {"type": "tool_result", "name": message.name,
                               "ok": "error" not in payload,
                               "detail": payload.get("error") or message.content[:200]}
                        if message.name == "make_chart" and payload.get("chart_path"):
                            chart_path = payload["chart_path"]
                            yield {"type": "chart", "path": chart_path}
                        for name in payload.get("cite_these_filenames") or []:
                            if name not in retrieved_files:
                                retrieved_files.append(name)
                    elif isinstance(message, AIMessage):
                        for call in message.tool_calls or []:
                            yield {"type": "tool_call", "name": call["name"], "args": call["args"]}
                        if not message.tool_calls and message.content:
                            answer = _ensure_citations(message.content, retrieved_files)
                            yield {"type": "answer", "text": answer}

        # LangChainTracer only records latest_run once the top-level run ends, so the
        # URL cannot be produced mid-stream - live progress is the event stream above.
        try:
            trace_url = tracer.get_run_url()
            yield {"type": "trace", "url": trace_url}
        except Exception:
            trace_url = None  # tracing disabled, or nothing was traced

    yield {"type": "done", "answer": answer, "steps": steps, "tools_used": tools_used,
           "chart_path": chart_path, "trace_url": trace_url}


def ask(question: str, session_id: str = "default") -> dict:
    """Run one question and return the result. Blocking wrapper around stream_events."""
    for event in stream_events(question, session_id):
        if event["type"] == "done":
            return {key: value for key, value in event.items() if key != "type"}
    return {"answer": "", "steps": 0, "tools_used": [], "chart_path": None}


if __name__ == "__main__":
    session = input("Document session [default]: ").strip() or "default"
    result = ask(input("Ask a question: "), session)
    print(f"\n{result['answer']}\n")
    print(f"[{result['steps']} model steps | tools: {', '.join(result['tools_used']) or 'none'}]")
    if result["chart_path"]:
        print(f"[chart: {result['chart_path']}]")
