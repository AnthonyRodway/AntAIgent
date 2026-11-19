"""Describe data files: columns, types, value ranges, empties and a few sample rows.

CSV, TSV and Parquet files are read with DuckDB. Excel sheets are read with openpyxl
and then profiled by DuckDB the same way. SQLite databases (including GeoPackages)
get a schema description. The description becomes the file's chunks, so a question
like "which file has the biomass column?" can find it.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from .chunking import Chunk

TABULAR_EXTENSIONS = frozenset({".csv", ".tsv", ".parquet", ".xlsx", ".xlsm"})
DATABASE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".gpkg"})
PROFILE_EXTENSIONS = TABULAR_EXTENSIONS | DATABASE_EXTENSIONS

MAX_EXCEL_ROWS = 100_000
SAMPLE_ABOVE_BYTES = 200_000_000  # bigger files are profiled from a random sample
SAMPLE_SIZE = 500_000
MAX_COLUMNS = 80
CATEGORICAL_MAX = 25
TOP_VALUES = 5
SAMPLE_ROWS = 5
MAX_CELL = 40
# A database could hold a view that never finishes (a recursive CTE, say), so each
# table or view gets this long before it's reported as unreadable.
SQLITE_SECONDS_PER_TABLE = 15

_FORMAT_NAMES = {".csv": "CSV", ".tsv": "TSV", ".parquet": "Parquet", ".xlsx": "Excel", ".xlsm": "Excel",
                 ".sqlite": "SQLite", ".sqlite3": "SQLite", ".db": "SQLite", ".gpkg": "GeoPackage"}
_GPKG_INTERNAL = ("gpkg_", "rtree_", "sqlite_")


class ProfileError(Exception):
    pass


@dataclass
class Column:
    name: str
    type: str
    null_pct: float | None = None
    distinct: int | None = None
    low: str | None = None
    high: str | None = None
    mean: str | None = None
    examples: list[str] = field(default_factory=list)
    top: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class Table:
    name: str  # sheet or table name; "" for single-table files
    rows: int | None
    columns: list[Column]
    header: list[str] = field(default_factory=list)
    sample: list[tuple] = field(default_factory=list)
    note: str = ""


# --- Reading ---------------------------------------------------------------------------


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise ProfileError("DuckDB isn't installed") from exc
    return duckdb.connect(config={
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        # Spill to the system temp folder; DuckDB's default is ./.tmp, which could be the project.
        "temp_directory": os.path.join(tempfile.gettempdir(), "antai-duckdb"),
    })


def _csv_source(con, path: Path) -> str:
    """Return a DuckDB table expression for a CSV, falling back to latin-1 for non-UTF-8 files."""
    p = path.as_posix().replace("'", "''")
    base = f"read_csv('{p}', sample_size=20480, null_padding=true, ignore_errors=true"
    for encoding in ("", ", encoding='latin-1'"):
        source = f"{base}{encoding})"
        try:
            con.execute(f"SELECT * FROM {source} LIMIT 1").fetchall()
            return source
        except Exception as exc:
            error = exc
    raise ProfileError(f"DuckDB couldn't read it: {str(error).splitlines()[0]}")


def _profile_relation(con, source: str, name: str, size: int, note: str = "") -> Table:
    if size > SAMPLE_ABOVE_BYTES:
        rows = con.execute(f"SELECT COUNT(*) FROM {source}").fetchall()[0][0]
        con.execute(f"CREATE OR REPLACE TEMP TABLE t AS SELECT * FROM {source} USING SAMPLE {SAMPLE_SIZE} ROWS")
        note = note or f"statistics from a random sample of {SAMPLE_SIZE:,} rows"
    else:
        con.execute(f"CREATE OR REPLACE TEMP TABLE t AS SELECT * FROM {source}")
        rows = con.execute("SELECT COUNT(*) FROM t").fetchall()[0][0]

    described = con.execute("DESCRIBE t").fetchall()
    names = [r[0] for r in described]
    table = Table(name, rows, [], names, note=note)
    if len(names) > MAX_COLUMNS:
        table.note = "; ".join(filter(None, [table.note, f"first {MAX_COLUMNS} of {len(names)} columns described"]))
    summary = {r[0]: r for r in con.execute("SUMMARIZE t").fetchall()} if rows else {}

    for col_name, col_type, *_ in described[:MAX_COLUMNS]:
        col = Column(col_name, col_type)
        s = summary.get(col_name)
        if s is not None:
            # SUMMARIZE columns: name, type, min, max, approx_unique, avg, std, q25, q50, q75, count, null_percentage
            col.low, col.high, col.distinct = s[2], s[3], s[4]
            col.mean = s[5] if _is_numeric(col_type) else None
            col.null_pct = float(s[11]) if s[11] is not None else None
        quoted = '"' + col_name.replace('"', '""') + '"'
        if rows and col_type in ("VARCHAR", "BOOLEAN") and (col.null_pct or 0) < 100:
            if col_type == "BOOLEAN" or (col.distinct is not None and col.distinct <= CATEGORICAL_MAX):
                col.top = [(str(v), c) for v, c in con.execute(
                    f"SELECT {quoted}, COUNT(*) AS n FROM t WHERE {quoted} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY n DESC, 1 LIMIT {TOP_VALUES}").fetchall()]
            else:
                col.examples = [str(r[0]) for r in con.execute(
                    f"SELECT DISTINCT {quoted} FROM t WHERE {quoted} IS NOT NULL LIMIT 3").fetchall()]
        table.columns.append(col)
    table.sample = con.execute(f"SELECT * FROM t LIMIT {SAMPLE_ROWS}").fetchall()
    return table


def _excel_tables(path: Path) -> list[Table]:
    try:
        import openpyxl
    except ImportError as exc:
        raise ProfileError("openpyxl isn't installed") from exc
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise ProfileError(f"openpyxl couldn't open it: {exc}") from exc
    tables: list[Table] = []
    con = _duckdb()
    try:
        with tempfile.TemporaryDirectory(prefix="antai-") as tmp:
            for sheet in workbook.worksheets:
                if not hasattr(sheet, "iter_rows"):
                    continue  # chart sheets
                if hasattr(sheet, "reset_dimensions"):
                    sheet.reset_dimensions()  # stored dimensions are often wrong
                tables.append(_excel_sheet(con, sheet, Path(tmp) / f"sheet{len(tables)}.csv"))
    finally:
        workbook.close()
        con.close()
    return tables


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return str(value)


def _excel_sheet(con, sheet, csv_path: Path) -> Table:
    rows = sheet.iter_rows(values_only=True)
    header: list | None = None
    for row in rows:
        if row and any(v is not None and str(v).strip() for v in row):
            header = list(row)
            break
    if header is None:
        return Table(sheet.title, 0, [], note="empty sheet")
    while header and (header[-1] is None or not str(header[-1]).strip()):
        header.pop()
    width = len(header)
    names: list[str] = []
    for i, value in enumerate(header, 1):
        name = str(value).strip() if value is not None and str(value).strip() else f"column_{i}"
        while name in names:
            name += "_"
        names.append(name)

    written = 0
    truncated = False
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in rows:
            if written >= MAX_EXCEL_ROWS:
                truncated = True
                break
            cells = [_cell(v) for v in list(row)[:width]]
            if any(cells):
                writer.writerow(cells + [""] * (width - len(cells)))
                written += 1
    if not written:
        return Table(sheet.title, 0, [Column(n, "") for n in names], names, note="header only")
    source = _csv_source(con, csv_path)
    note = f"first {MAX_EXCEL_ROWS:,} rows profiled" if truncated else ""
    return _profile_relation(con, source, sheet.title, csv_path.stat().st_size, note)


def _sqlite_tables(path: Path) -> list[Table]:
    with open(path, "rb") as fh:
        if fh.read(16) != b"SQLite format 3\x00":
            raise ProfileError("not a SQLite database")
    uri = "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        # The file may come from anywhere: don't let its views and triggers call functions.
        conn.execute("PRAGMA trusted_schema = OFF")
    except sqlite3.Error as exc:
        raise ProfileError(f"SQLite couldn't open it: {exc}") from exc
    deadline = [0.0]
    conn.set_progress_handler(lambda: time.monotonic() > deadline[0], 10_000)  # non-zero aborts the query
    tables: list[Table] = []
    try:
        deadline[0] = time.monotonic() + SQLITE_SECONDS_PER_TABLE
        objects = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name"
        ).fetchall()
        for name, kind in objects:
            if name.lower().startswith(_GPKG_INTERNAL):
                continue
            quoted = '"' + name.replace('"', '""') + '"'
            deadline[0] = time.monotonic() + SQLITE_SECONDS_PER_TABLE
            try:
                info = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
                count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
                sample = conn.execute(f"SELECT * FROM {quoted} LIMIT 3").fetchall()
            except sqlite3.Error as exc:
                timed_out = "interrupted" in str(exc)
                reason = f"took over {SQLITE_SECONDS_PER_TABLE}s to read" if timed_out else f"couldn't read: {exc}"
                tables.append(Table(name, None, [], note=reason))
                continue
            columns = [Column(r[1], r[2] or "") for r in info]
            tables.append(Table(name, count, columns, [c.name for c in columns], sample, note="view" if kind == "view" else ""))
    finally:
        conn.close()
    return tables


def profile_file(path: Path) -> list[Table]:
    ext = path.suffix.lower()
    if ext in DATABASE_EXTENSIONS:
        return _sqlite_tables(path)
    if ext in (".xlsx", ".xlsm"):
        return _excel_tables(path)
    con = _duckdb()
    try:
        if ext == ".parquet":
            source = "read_parquet('" + path.as_posix().replace("'", "''") + "')"
        else:
            source = _csv_source(con, path)
        try:
            return [_profile_relation(con, source, "", os.path.getsize(path))]
        except ProfileError:
            raise
        except Exception as exc:
            raise ProfileError(f"DuckDB couldn't profile it: {str(exc).splitlines()[0]}") from exc
    finally:
        con.close()


# --- Describing ------------------------------------------------------------------------


def _is_numeric(type_name: str) -> bool:
    t = type_name.upper()
    return any(w in t for w in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC", "HUGEINT"))


def _friendly_type(type_name: str) -> str:
    t = type_name.upper()
    if not t:
        return "unknown"
    if "INT" in t:
        return "integer"
    if _is_numeric(t):
        return "decimal"
    if t in ("VARCHAR", "TEXT") or t.startswith(("VARCHAR", "CHAR", "TEXT")):
        return "text"
    if t.startswith("TIMESTAMP") or t == "DATETIME":
        return "timestamp"
    return t.lower()


def _number(value) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}"
    if abs(x) >= 1:
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    return f"{x:.4g}"


def _short(value) -> str:
    text = _cell(value).replace("\n", " ")
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 3] + "..."


def _describe_column(col: Column) -> str:
    kind = _friendly_type(col.type)
    if col.null_pct is not None and col.null_pct >= 100:
        return f"- {col.name}: always empty"
    parts = [kind]
    if col.top:
        values = ", ".join(f"{_short(v)} ({n:,})" for v, n in col.top)
        more = f" of {col.distinct:,}" if col.distinct and col.distinct > len(col.top) else ""
        parts.append(f"values{more}: {values}")
    else:
        if col.distinct is not None and kind == "text":
            parts.append(f"~{col.distinct:,} distinct")
        if col.low is not None and col.high is not None and kind != "text":
            if _is_numeric(col.type):
                parts.append(f"{_number(col.low)} to {_number(col.high)}")
            else:
                parts.append(f"{_short(col.low)} to {_short(col.high)}")
        if col.mean is not None:
            parts.append(f"mean {_number(col.mean)}")
        if col.examples:
            parts.append("e.g. " + ", ".join(_short(e) for e in col.examples))
    if col.null_pct:
        parts.append(f"{col.null_pct:g}% empty")
    return f"- {col.name}: " + ", ".join(parts)


def describe(rel: str, table: Table, n_tables: int) -> tuple[str, list[str]]:
    """Return (header line, body lines) describing one table."""
    fmt = _FORMAT_NAMES.get(PurePosixPath(rel).suffix.lower(), "Data")
    what = {"Excel": "sheet", "SQLite": "table", "GeoPackage": "table"}.get(fmt, "table")
    where = f", {what} {table.name!r}" if table.name and (n_tables > 1 or fmt in ("Excel", "SQLite", "GeoPackage")) else ""
    size = f"{table.rows:,} rows" if table.rows is not None else "unknown rows"
    header = f"Data file: {rel} ({fmt}{where}), {size}, {len(table.header) or len(table.columns)} columns"
    lines: list[str] = []
    if table.note:
        lines.append(f"Note: {table.note}")
    if table.columns:
        lines.append("Columns:")
        lines.extend(_describe_column(c) for c in table.columns)
    if table.sample:
        lines.append("Sample rows:")
        cols = table.header[: len(table.columns) or None]
        lines.append(" | ".join(cols))
        for row in table.sample:
            cells = ["<binary>" if isinstance(v, (bytes, memoryview)) else _short(v) for v in row[: len(cols)]]
            lines.append(" | ".join(cells))
    return header, lines


def profile_chunks(rel: str, path: Path, max_chars: int) -> list[Chunk]:
    """Profile a data file into chunks; each starts with a line naming the file and table."""
    tables = profile_file(path)
    chunks: list[Chunk] = []
    for number, table in enumerate(tables, 1):
        header, lines = describe(rel, table, len(tables))
        symbols = ([table.name] if table.name else []) + [c.name for c in table.columns][:40]
        groups: list[list[str]] = [[]]
        size = len(header)
        for line in lines:
            if groups[-1] and size + len(line) + 1 > max_chars:
                groups.append([])
                size = len(header)
            groups[-1].append(line)
            size += len(line) + 1
        chunks.extend(Chunk("\n".join([header, *g]), number, number, "data", list(symbols)) for g in groups)
    return chunks
