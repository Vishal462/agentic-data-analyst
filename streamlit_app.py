"""Streamlit frontend for the analyst.
    streamlit run streamlit_app.py
Renders the agent's progress live, because a question takes minutes and a UI that
shows nothing until the end reads as broken. Consumes app.agent.stream_events, so
nothing here knows about LangGraph.
"""
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="wide")

TOOL_LABELS = {
    "list_tables": "Listing tables",
    "describe_table": "Reading the schema",
    "run_sql": "Querying the database",
    "compare_groups": "Comparing groups",
    "correlate": "Computing correlations",
    "find_outliers": "Looking for outliers",
    "describe_distribution": "Summarising the distribution",
    "make_chart": "Drawing the chart",
    "search_documents": "Searching your documents",
}
PAGE_SIZES = [50, 100, 250, 500]
NEW_SESSION = "+ New session..."


@st.cache_resource(show_spinner="Loading the model and database...")
def _load():
    """Imported once per process; module-level model and DB setup is expensive."""
    from langsmith.utils import tracing_is_enabled

    from app.agent import stream_events
    from app.llm import MODEL
    return stream_events, MODEL, tracing_is_enabled(), os.getenv("LANGSMITH_PROJECT", "default")


stream_events, model_name, tracing_on, langsmith_project = _load()

# Read fresh every rerun rather than cached: an upload changes the active database,
# and a stale table list here would leave the Data tab pointing at the old dataset.
from app.db import SUPPORTED_DATASET_SUFFIXES, get_active_dataset, register_dataset, use_demo_dataset
from app.rag.vector_store import delete_session

active = get_active_dataset()
tables, db_name = active["tables"], active["database"]


def _no_trace_caption() -> str:
    """Say what is actually wrong; 'tracing may be off' is misleading when it is on."""
    if tracing_on:
        return (f"No trace link for this run. Tracing is on (project `{langsmith_project}`), so the run is "
                f"in LangSmith - only the direct link is missing. If this happens for every run, restart "
                f"the Streamlit server: it does not reload Python modules while running.")
    return "No trace link: LangSmith tracing is off (set LANGSMITH_TRACING=true in .env and restart)."

st.session_state.setdefault("history", [])
st.session_state.setdefault("runs", [])
st.session_state.setdefault("doc_version", 0)   # bumped to invalidate session/doc caches


# --- Data tab queries -------------------------------------------------------
# st.tabs renders every tab on each rerun, so these must be cached or the Data
# tab would re-query on every chat message.

@st.cache_data(show_spinner=False)
def _table_shape(table: str) -> tuple[int, list[str]]:
    from app.db import query_df, quote_identifier
    rows = int(query_df(f"SELECT COUNT(*) AS n FROM {quote_identifier(table)}").iloc[0, 0])
    columns = [str(name) for name in query_df(f"DESCRIBE {quote_identifier(table)}").iloc[:, 0]]
    return rows, columns


def _where(filter_column: str | None, filter_text: str) -> tuple[str, dict]:
    from app.db import quote_identifier
    if filter_column and filter_text:
        return (f"WHERE CAST({quote_identifier(filter_column)} AS VARCHAR) ILIKE :pattern",
                {"pattern": f"%{filter_text}%"})
    return "", {}


@st.cache_data(show_spinner=False)
def _match_count(table: str, filter_column: str | None, filter_text: str) -> int | None:
    """Rows matching the filter, so page count is known before fetching a page."""
    from app.db import query_df, quote_identifier
    where, params = _where(filter_column, filter_text)
    if not where:
        return None
    return int(query_df(f"SELECT COUNT(*) AS n FROM {quote_identifier(table)} {where}", params).iloc[0, 0])


@st.cache_data(show_spinner=False)
def _page_of(table: str, offset: int, size: int, sort_column: str | None, descending: bool,
             filter_column: str | None, filter_text: str):
    """One page, fetched server-side. Sorting is done in SQL, never in the browser:
    sorting a page client-side would order 100 rows and look like it ordered the table."""
    from app.db import query_df, quote_identifier
    where, params = _where(filter_column, filter_text)
    order = f"ORDER BY {quote_identifier(sort_column)} {'DESC' if descending else 'ASC'}" if sort_column else ""
    return query_df(f"SELECT * FROM {quote_identifier(table)} {where} {order} "
                    f"LIMIT {int(size)} OFFSET {int(offset)}", params or None)


def _sessions() -> dict[str, int]:
    """Uncached at ~13ms: a cached list would hide sessions created by the CLI or
    another window, and a stale session list is worse than the query cost."""
    from app.rag.vector_store import list_sessions
    return list_sessions()


@st.cache_data(show_spinner=False)
def _documents(session_id: str, _bust: int) -> dict[str, list[dict]]:
    """Chunk text per document, so the Documents tab can show what was indexed."""
    from app.rag.vector_store import session_chunks
    return session_chunks(session_id)


with st.sidebar:
    st.subheader("Setup")
    origin = "uploaded" if active["is_upload"] else "demo"
    st.caption(f"**Model** `{model_name}`  \n**Database** `{db_name}` ({origin})  "
               f"\n**Tables** {', '.join(tables) or 'none'}")
    # Sessions persist on disk, so past ones are listed rather than remembered by the
    # user; free text alone made a typo silently create a new empty session.
    sessions = _sessions()
    # Default to the session that ships with the demo data, not just the first alphabetically.
    st.session_state.setdefault("session_id",
                                "flights-demo" if "flights-demo" in sessions
                                else next(iter(sessions), "flights-demo"))
    options = sorted(sessions)
    if st.session_state.session_id not in options:
        options.append(st.session_state.session_id)
    options.append(NEW_SESSION)
    choice = st.selectbox("Document session", options, index=options.index(st.session_state.session_id),
                          format_func=lambda name: (name if name == NEW_SESSION
                                                    else f"{name}  ({sessions.get(name, 0)} chunks)"),
                          help="Documents are indexed per session. Pick a past session to use its "
                               "documents again, or create a new one.")
    if choice == NEW_SESSION:
        typed = st.text_input("New session name", placeholder="e.g. q3-reports").strip()
        session = typed or st.session_state.session_id
        if typed:
            st.session_state.session_id = typed
    else:
        session = choice
        st.session_state.session_id = choice

    if session in sessions:
        # Ticking the confirm box reruns the script, which would otherwise collapse this
        # expander and hide the button the tick just enabled.
        with st.expander("Delete this session", expanded=st.session_state.get(f"confirm_{session}", False)):
            st.caption(f"Removes the {sessions[session]} indexed chunk(s) for `{session}`. "
                       f"Only the index is deleted - your original files are untouched.")
            confirmed = st.checkbox(f"Yes, delete '{session}'", key=f"confirm_{session}")
            if st.button("Delete permanently", disabled=not confirmed, use_container_width=True):
                delete_session(session)
                remaining = [name for name in sorted(sessions) if name != session]
                st.session_state.session_id = remaining[0] if remaining else "flights-demo"
                st.session_state.doc_version += 1
                st.cache_data.clear()
                st.rerun()
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.runs = []
        st.rerun()

st.title("Agentic Data Analyst")
st.caption("Ask about the data, or about your uploaded documents. Every number comes from a real query.")

tab_analyze, tab_data, tab_docs, tab_runs = st.tabs(["Analyze", "Data", "Documents", "Runs"])

# --- Analyze ----------------------------------------------------------------
with tab_analyze:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("chart"):
                st.image(turn["chart"])
            if turn.get("footer"):
                st.caption(turn["footer"])

# --- Data -------------------------------------------------------------------
with tab_data:
    with st.expander("Upload a dataset", expanded=not tables):
        st.caption("CSV, TSV, Excel, or Parquet. The upload becomes the active dataset for "
                   "your questions and replaces any previously uploaded one. "
                   "This is structured data - documents for retrieval go in the Documents tab.")
        dataset_file = st.file_uploader(
            "Dataset file", type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_DATASET_SUFFIXES)],
            key="dataset_upload", accept_multiple_files=False)
        upload_row = st.columns([1, 1, 3])
        if upload_row[0].button("Use this dataset", disabled=dataset_file is None):
            with st.spinner("Registering in DuckDB..."):
                target = Path(tempfile.mkdtemp()) / dataset_file.name
                target.write_bytes(dataset_file.getbuffer())
                try:
                    info = register_dataset(target)
                    st.cache_data.clear()          # table shape, page and filter caches
                    st.session_state.data_page = 1
                    st.success(f"Loaded **{info['rows']:,} rows x {len(info['columns'])} columns** "
                               f"into table `{info['table']}` from {info['source']}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load {dataset_file.name}: {exc}")
                    st.caption("The previously active dataset is unchanged.")
        if active["is_upload"] and upload_row[1].button("Back to demo data"):
            use_demo_dataset()
            st.cache_data.clear()
            st.session_state.data_page = 1
            st.rerun()

    if not tables:
        st.info("No tables in this database. Upload a dataset above.")
    else:
        table = st.selectbox("Table", tables) if len(tables) > 1 else tables[0]
        total_rows, columns = _table_shape(table)

        with st.form("data_filter"):
            filter_row = st.columns([2, 3, 2, 2])
            filter_column = filter_row[0].selectbox("Filter column", ["(none)"] + columns)
            filter_text = filter_row[1].text_input("Contains")
            sort_column = filter_row[2].selectbox("Sort by", ["(unsorted)"] + columns)
            direction = filter_row[3].selectbox("Order", ["Descending", "Ascending"])
            st.form_submit_button("Apply", use_container_width=False)

        filter_column = None if filter_column == "(none)" else filter_column
        sort_column = None if sort_column == "(unsorted)" else sort_column

        page_size = st.selectbox("Rows per page", PAGE_SIZES, index=1)

        # Any change to what is being viewed sends you back to page 1, otherwise you
        # can land on a page that no longer exists after filtering.
        signature = (table, filter_column, filter_text, sort_column, direction, page_size)
        if st.session_state.get("data_signature") != signature:
            st.session_state.data_signature = signature
            st.session_state.data_page = 1

        matched = _match_count(table, filter_column, filter_text)
        shown_total = matched if matched is not None else total_rows
        pages = max(1, -(-shown_total // page_size))
        page = min(st.session_state.get("data_page", 1), pages)
        st.session_state.data_page = page

        summary = st.columns(4)
        summary[0].metric("Rows", f"{total_rows:,}")
        summary[1].metric("Columns", len(columns))
        summary[2].metric("Matching filter", f"{shown_total:,}" if matched is not None else "-")
        summary[3].metric("Page", f"{page:,} of {pages:,}")

        frame = _page_of(table, (page - 1) * page_size, page_size,
                         sort_column, direction == "Descending", filter_column, filter_text)
        if frame.empty:
            st.info("No rows match that filter.")
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)

        def _go(target: int) -> None:
            st.session_state.data_page = max(1, min(target, pages))

        nav = st.columns([1, 1, 3, 1, 1])
        nav[0].button("« First", use_container_width=True, disabled=page <= 1,
                      on_click=_go, args=(1,))
        nav[1].button("‹ Previous", use_container_width=True, disabled=page <= 1,
                      on_click=_go, args=(page - 1,))
        nav[2].markdown(f"<div style='text-align:center;padding-top:0.45rem'>"
                        f"Showing rows {(page - 1) * page_size + 1:,}-{min(page * page_size, shown_total):,} "
                        f"of {shown_total:,}</div>", unsafe_allow_html=True)
        nav[3].button("Next ›", use_container_width=True, disabled=page >= pages,
                      on_click=_go, args=(page + 1,))
        nav[4].button("Last »", use_container_width=True, disabled=page >= pages,
                      on_click=_go, args=(pages,))

        st.caption("Rows are paged and sorted in DuckDB, so sorting orders the whole table, "
                   "not just this page.")

# --- Documents --------------------------------------------------------------
with tab_docs:
    documents = _documents(session, st.session_state.doc_version)
    if documents:
        total = sum(len(chunks) for chunks in documents.values())
        st.write(f"**{total} chunk(s)** indexed for session `{session}`. "
                 f"Open a document to read exactly what was indexed:")
        for name, chunks in documents.items():
            with st.expander(f"{name}  -  {len(chunks)} chunk(s)"):
                for position, chunk in enumerate(chunks, start=1):
                    label = chunk["chunk_id"] if chunk["chunk_id"] is not None else position
                    st.caption(f"Chunk {label}  -  {len(chunk['text'])} characters")
                    st.markdown(chunk["text"] or "_(empty)_")
                    if position < len(chunks):
                        st.divider()
    else:
        st.info(f"No documents indexed for session `{session}` yet.")

    uploads = st.file_uploader("PDF, Word, Markdown, or text", type=["pdf", "docx", "md", "txt"],
                               accept_multiple_files=True)
    if uploads and st.button("Index documents"):
        from app.rag.ingestion import ingest_documents
        with st.spinner("Indexing..."):
            folder = Path(tempfile.mkdtemp())
            paths = []
            for upload in uploads:
                target = folder / upload.name
                target.write_bytes(upload.getbuffer())
                paths.append(target)
            try:
                result = ingest_documents(paths, session)
                st.session_state.doc_version += 1
                st.success(f"Indexed {result['chunks_indexed']} new chunk(s) from "
                           f"{', '.join(result['documents'])}; skipped {result['chunks_skipped']} duplicate(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")

# --- Runs -------------------------------------------------------------------
with tab_runs:
    if not st.session_state.runs:
        st.info("No questions asked yet this session.")
    else:
        st.caption(f"{len(st.session_state.runs)} run(s) this session. "
                   "Open a trace to see model calls, tool calls, timings, and errors.")
        for run in reversed(st.session_state.runs):
            with st.expander(f"{run['question'][:90]}  -  {run['seconds']:.0f}s, {run['steps']} steps"):
                st.write(f"**Tools:** {', '.join(run['tools']) or 'none'}")
                if run.get("trace"):
                    st.markdown(f"[Open this run in LangSmith]({run['trace']})")
                else:
                    st.caption(_no_trace_caption())
                if run.get("chart"):
                    st.image(run["chart"])


def _run_agent(question: str) -> dict:
    """Stream the agent, rendering each step as it happens."""
    result = {"answer": "", "chart": None, "trace": None, "steps": 0, "tools": [], "seconds": 0.0}
    start = time.perf_counter()
    with st.status("Thinking...", expanded=True) as status:
        for event in stream_events(question, session):
            elapsed = time.perf_counter() - start
            kind = event["type"]
            if kind == "trace":
                result["trace"] = event["url"]
                st.caption(f"[Open this run in LangSmith]({event['url']})")
            elif kind == "tool_call":
                label = TOOL_LABELS.get(event["name"], event["name"])
                status.update(label=f"{label}...")
                st.write(f"**{label}**")
                args = event["args"]
                if event["name"] == "run_sql" and args.get("query"):
                    st.code(args["query"], language="sql")
                elif args:
                    st.code(json.dumps(args, default=str), language="json")
            elif kind == "tool_result":
                if not event["ok"]:
                    st.write(f"⚠️ `{event['name']}` failed - retrying: {event['detail'][:200]}")
                status.update(label="Generating final answer...")
            elif kind == "chart":
                result["chart"] = event["path"]
            elif kind == "done":
                result.update(answer=event["answer"] or "_The agent stopped without producing an answer._",
                              steps=event["steps"], tools=event["tools_used"], seconds=elapsed)
                status.update(label=f"Done in {elapsed:.0f}s", state="complete", expanded=False)
    return result


if question := st.chat_input("Ask a question about the data..."):
    with tab_analyze:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                result = _run_agent(question)
            except Exception as exc:
                result = {"answer": f"**{type(exc).__name__}:** {exc}", "chart": None,
                          "trace": None, "steps": 0, "tools": [], "seconds": 0.0}
    footer = (f"{result['seconds']:.0f}s · {result['steps']} model steps · "
              f"tools: {', '.join(result['tools']) or 'none'}"
              + (f" · [LangSmith trace]({result['trace']})" if result["trace"] else ""))
    st.session_state.history.append({"role": "assistant", "content": result["answer"],
                                     "chart": result["chart"], "footer": footer})
    st.session_state.runs.append({"question": question, **result})
    st.rerun()  # so the Runs tab picks up the run that just finished
