import datetime as dt
import sqlite3

import pytest

from antaigent.ingest.profiling import ProfileError, profile_chunks

duckdb = pytest.importorskip("duckdb")


def write_csv(path, rows=300):
    lines = ["site,species,biomass,planted,unused"]
    for i in range(rows):
        biomass = "" if i % 10 == 0 else f"{i * 1.5}"
        lines.append(f"s{i},{'pine' if i % 3 else 'spruce'},{biomass},2024-01-{1 + i % 28:02d},")
    path.write_text("\n".join(lines) + "\n")


def test_csv_profile_describes_columns_and_values(tmp_path):
    write_csv(tmp_path / "plots.csv")
    (chunk,) = profile_chunks("data/plots.csv", tmp_path / "plots.csv", 4000)
    text = chunk.text
    assert chunk.kind == "data"
    assert chunk.symbols == ["site", "species", "biomass", "planted", "unused"]
    assert text.startswith("Data file: data/plots.csv (CSV), 300 rows, 5 columns")
    assert "- species: text, values: pine (200), spruce (100)" in text
    assert "- biomass: decimal, 1.5 to 448.5" in text and "10% empty" in text
    assert "- planted: date, 2024-01-01 to 2024-01-28" in text
    assert "- unused: always empty" in text
    assert "s1 | pine | 1.5 | 2024-01-02" in text


def test_parquet_profile(tmp_path):
    write_csv(tmp_path / "plots.csv")
    target = (tmp_path / "plots.parquet").as_posix()
    duckdb.connect().execute(f"COPY (SELECT * FROM read_csv('{(tmp_path / 'plots.csv').as_posix()}')) TO '{target}' (FORMAT parquet)")
    (chunk,) = profile_chunks("plots.parquet", tmp_path / "plots.parquet", 4000)
    assert "(Parquet), 300 rows" in chunk.text and "species" in chunk.symbols


def test_excel_sheets_become_separate_chunks(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Stands"
    ws.append([])  # a blank first row: the header is the first non-empty row
    ws.append(["Stand ID", "Area (ha)", "Planted"])
    for i in range(12):
        ws.append([f"ST-{i}", 2.5 * i, dt.date(2020, 1, 1 + i)])
    wb.create_sheet("Notes")
    wb.save(tmp_path / "stands.xlsx")

    chunks = profile_chunks("stands.xlsx", tmp_path / "stands.xlsx", 4000)
    assert [c.start_line for c in chunks] == [1, 2]
    assert "sheet 'Stands'), 12 rows, 3 columns" in chunks[0].text
    assert chunks[0].symbols[:2] == ["Stands", "Stand ID"]
    assert "- Area (ha): decimal, 0 to 27.5" in chunks[0].text
    assert "empty sheet" in chunks[1].text


def test_geopackage_schema_skips_internal_tables(tmp_path):
    path = tmp_path / "sites.gpkg"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE gpkg_contents (table_name TEXT)")
    conn.execute("CREATE TABLE plots (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT)")
    conn.execute("INSERT INTO plots (geom, name) VALUES (x'0001', 'north')")
    conn.commit()
    conn.close()
    (chunk,) = profile_chunks("sites.gpkg", path, 4000)
    assert "(GeoPackage, table 'plots'), 1 rows, 3 columns" in chunk.text
    assert "gpkg_contents" not in chunk.text
    assert "1 | <binary> | north" in chunk.text


def test_a_view_that_never_finishes_is_given_up_on(tmp_path, monkeypatch):
    monkeypatch.setattr("antaigent.ingest.profiling.SQLITE_SECONDS_PER_TABLE", 0.5)
    path = tmp_path / "trap.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sites (name TEXT)")
    conn.execute("INSERT INTO sites VALUES ('north')")
    conn.execute(
        "CREATE VIEW forever AS WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT x FROM n"
    )
    conn.commit()
    conn.close()
    chunks = profile_chunks("trap.sqlite", path, 4000)
    text = "\n".join(c.text for c in chunks)
    assert "table 'sites'), 1 rows" in text
    assert "table 'forever'), unknown rows" in text and "took over 0.5s to read" in text


def test_non_sqlite_db_is_rejected(tmp_path):
    (tmp_path / "Thumbs.db").write_bytes(b"\xd0\xcf\x11\xe0 not sqlite")
    with pytest.raises(ProfileError):
        profile_chunks("Thumbs.db", tmp_path / "Thumbs.db", 4000)
