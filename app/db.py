"""Dataset-agnostic DuckDB access and metadata discovery."""
import os
import re
from pathlib import Path
from typing import Any
import duckdb
import pandas as pd
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "analytics_test.duckdb"  # Demo fixture.
DB_PATH = Path(os.getenv("DATA_ANALYST_DB_PATH") or DEFAULT_DB_PATH)
# Uploaded datasets live in their own file so the demo database is never written to.
# Overridable so a test run does not contend with a running app for the same file.
UPLOAD_DB_PATH = Path(os.getenv("DATA_ANALYST_UPLOAD_DB_PATH") or PROJECT_ROOT / "data" / "uploads.duckdb")
SUPPORTED_DATASET_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}

# Read-only at the engine level: a stronger guarantee than pattern-matching the
# SQL, and it lets the agent skip a hand-rolled statement guard entirely. Writes
# happen only in register_dataset, through a separate short-lived connection.
engine = create_engine(f"duckdb:///{DB_PATH}", connect_args={"read_only": True})
_active_path = DB_PATH


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as connection:
        return pd.read_sql(text(sql), connection, params=params)


def discover_tables() -> list[str]:
    return inspect(engine).get_table_names()


# --- Active dataset ---------------------------------------------------------

def _use_database(path: Path) -> None:
    """Point every later query at `path`, read-only. DuckDB refuses two connections
    with different settings to one file, so the old engine is disposed first."""
    global engine, _active_path
    engine.dispose()
    engine = create_engine(f"duckdb:///{path}", connect_args={"read_only": True})
    _active_path = Path(path)


def safe_table_name(filename: str) -> str:
    """A quoted-safe identifier derived from a filename; never interpolated raw."""
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}" if cleaned else "dataset"
    return cleaned[:60]


def _load_into(connection, table: str, path: Path) -> None:
    """CSV and Parquet are read by DuckDB itself, so types are inferred properly and
    the file never passes through pandas. Excel has no native reader, so it does."""
    suffix = path.suffix.lower()
    target = quote_identifier(table)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        connection.execute(f"CREATE TABLE {target} AS SELECT * FROM "
                           f"read_csv_auto(?, header = true, sample_size = 100000, delim = ?)",
                           [str(path), delimiter])
    elif suffix == ".parquet":
        connection.execute(f"CREATE TABLE {target} AS SELECT * FROM read_parquet(?)", [str(path)])
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
        if frame.empty or not len(frame.columns):
            raise ValueError("The spreadsheet parsed successfully but contains no rows or no columns.")
        frame.columns = [str(name).strip() or f"column_{index}" for index, name in enumerate(frame.columns)]
        connection.register("_incoming", frame)
        try:
            connection.execute(f"CREATE TABLE {target} AS SELECT * FROM _incoming")
        finally:
            connection.unregister("_incoming")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of: "
                         f"{', '.join(sorted(SUPPORTED_DATASET_SUFFIXES))}.")


def register_dataset(source: str | Path, table_name: str | None = None) -> dict[str, Any]:
    """Load a tabular file into DuckDB and make it the active dataset.
    Uploading replaces any previously uploaded table, so exactly one uploaded
    dataset is active at a time and the agent's schema stays unambiguous. The
    demo database is untouched and is restored by use_demo_dataset()."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() not in SUPPORTED_DATASET_SUFFIXES:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use one of: "
                         f"{', '.join(sorted(SUPPORTED_DATASET_SUFFIXES))}.")
    table = table_name or safe_table_name(path.name)
    staging = f"_staging_{table}"[:63]
    UPLOAD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A short-lived write connection; nothing else may hold the file open.
    if _active_path == UPLOAD_DB_PATH:
        engine.dispose()
    try:
        connection = duckdb.connect(str(UPLOAD_DB_PATH))
    except duckdb.IOException as exc:
        # DuckDB allows one writer per file, so another running app blocks the load.
        if _active_path == UPLOAD_DB_PATH:
            _use_database(UPLOAD_DB_PATH)
        raise RuntimeError(
            f"Could not open {UPLOAD_DB_PATH.name} for writing - another process is using it. "
            f"Close any other running copy of the app and try again."
        ) from exc
    try:
        # Load into a staging table first, so a parse failure never destroys the
        # dataset that is currently active.
        connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(staging)}")
        try:
            _load_into(connection, staging, path)
            rows, columns = _describe(connection, staging)
            if not rows or not columns:
                raise ValueError("The file parsed successfully but contains no rows or no columns.")
            for existing in [row[0] for row in connection.execute("SHOW TABLES").fetchall()]:
                if existing != staging:
                    connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(existing)}")
            connection.execute(f"ALTER TABLE {quote_identifier(staging)} RENAME TO {quote_identifier(table)}")
        except Exception:
            connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(staging)}")
            raise
    finally:
        connection.close()
        if _active_path == UPLOAD_DB_PATH:
            _use_database(UPLOAD_DB_PATH)  # reopen the engine disposed above
    _use_database(UPLOAD_DB_PATH)
    return {"table": table, "rows": rows, "columns": columns,
            "source": path.name, "database": str(UPLOAD_DB_PATH)}


def _describe(connection, table: str) -> tuple[int, list[str]]:
    rows = int(connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0])
    columns = [row[0] for row in connection.execute(f"DESCRIBE {quote_identifier(table)}").fetchall()]
    return rows, columns


def use_demo_dataset() -> dict[str, Any]:
    """Switch back to the built-in demo database."""
    _use_database(DB_PATH)
    return get_active_dataset()


def get_active_dataset() -> dict[str, Any]:
    tables = discover_tables()
    return {"database": _active_path.name, "path": str(_active_path), "tables": tables,
            "is_upload": _active_path == UPLOAD_DB_PATH,
            "table": tables[0] if len(tables) == 1 else None}


def schema_signature() -> tuple:
    """Cheap identity of the current schema; changes when the active dataset changes."""
    return (str(_active_path), tuple(discover_tables()))


def _column_kind(type_name: str) -> str:
    upper = type_name.upper()
    if any(token in upper for token in ("INT", "DECIMAL", "NUMERIC", "DOUBLE", "FLOAT", "REAL")):
        return "numeric"
    if any(token in upper for token in ("DATE", "TIME", "TIMESTAMP")):
        return "date"
    if "BOOL" in upper:
        return "boolean"
    return "categorical"


def dataset_profile(table: str, sample_limit: int = 5) -> dict[str, Any]:
    """Return validated schema, counts, missingness, and small sample values."""
    if table not in discover_tables():
        raise ValueError(f"Unknown table: {table}")
    table_sql = quote_identifier(table)
    description = query_df(f"DESCRIBE {table_sql}")
    row_count = int(query_df(f"SELECT COUNT(*) AS n FROM {table_sql}").iloc[0, 0])
    columns = []
    for item in description.itertuples(index=False):
        name, data_type = item[0], str(item[1])
        column_sql = quote_identifier(name)
        missing = int(query_df(f"SELECT COUNT(*) - COUNT({column_sql}) AS n FROM {table_sql}").iloc[0, 0])
        values = query_df(f"SELECT DISTINCT {column_sql} AS value FROM {table_sql} WHERE {column_sql} IS NOT NULL LIMIT {int(sample_limit)}")["value"].tolist()
        columns.append({"name": name, "data_type": data_type, "kind": _column_kind(data_type), "missing_count": missing, "sample_values": [str(value) for value in values]})
    return {"table": table, "row_count": row_count, "column_count": len(columns), "columns": columns,
            "numeric_columns": [c["name"] for c in columns if c["kind"] == "numeric"],
            "categorical_columns": [c["name"] for c in columns if c["kind"] == "categorical"],
            "date_columns": [c["name"] for c in columns if c["kind"] == "date"]}


def database_catalog() -> dict[str, dict[str, Any]]:
    return {table: dataset_profile(table) for table in discover_tables()}
