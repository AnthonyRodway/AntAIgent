"""PDFs are read page by page, with their filled-in form fields, and cited by page."""

import io
import os

import pytest

pypdf = pytest.importorskip("pypdf")
from pypdf import PdfWriter  # noqa: E402
from pypdf.generic import (  # noqa: E402
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    TextStringObject,
)

from antaigent.index.indexer import _read, learn  # noqa: E402
from antaigent.index.store import FileRecord, Store  # noqa: E402
from antaigent.ingest.chunking import Chunker  # noqa: E402
from antaigent.ingest.pdf import PdfError, read_pdf  # noqa: E402
from antaigent.ingest.walker import SourceFile  # noqa: E402
from antaigent.query.answer import format_hit  # noqa: E402
from antaigent.query.retrieval import search  # noqa: E402


def _name(value: str) -> NameObject:
    return NameObject(value)


def make_pdf(pages: list[str], fields: dict[int, list[dict]] | None = None, password: str | None = None) -> bytes:
    """A PDF whose pages show the given lines of text in Helvetica. `fields` maps a page
    index to form field widgets, given as their PDF dictionary entries."""
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({
        _name("/Type"): _name("/Font"), _name("/Subtype"): _name("/Type1"), _name("/BaseFont"): _name("/Helvetica"),
    }))
    acro_fields = ArrayObject()
    for index, text in enumerate(pages):
        page = writer.add_blank_page(612, 792)
        if text:
            ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
            for line in text.split("\n"):
                escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                ops.append(f"({escaped}) Tj T*")
            ops.append("ET")
            stream = DecodedStreamObject()
            stream.set_data("\n".join(ops).encode("latin-1"))
            page[_name("/Contents")] = writer._add_object(stream)
            page[_name("/Resources")] = DictionaryObject({_name("/Font"): DictionaryObject({_name("/F1"): font})})
        annots = ArrayObject()
        for entries in (fields or {}).get(index, []):
            widget = DictionaryObject({_name("/Type"): _name("/Annot"), _name("/Subtype"): _name("/Widget"),
                                       _name("/Rect"): ArrayObject()})
            for key, value in entries.items():
                widget[_name(key)] = value
            ref = writer._add_object(widget)
            annots.append(ref)
            acro_fields.append(ref)
        if annots:
            page[_name("/Annots")] = annots
    if acro_fields:
        writer._root_object[_name("/AcroForm")] = DictionaryObject({_name("/Fields"): acro_fields})
    if password:
        writer.encrypt(user_password=password, algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def text_field(name: str, value: str) -> dict:
    return {"/FT": _name("/Tx"), "/T": TextStringObject(name), "/V": TextStringObject(value)}


def checkbox(name: str, on: bool) -> dict:
    return {"/FT": _name("/Btn"), "/T": TextStringObject(name), "/V": _name("/Yes" if on else "/Off")}


def test_pages_and_filled_form_fields_are_read():
    radio = DictionaryObject({_name("/FT"): _name("/Btn"), _name("/T"): TextStringObject("Season"),
                              _name("/V"): _name("/Spring")})
    data = make_pdf(
        ["Planting Activity Form\nSite name:", "", "Species planted\nAcer rubrum 25"],
        fields={
            0: [text_field("Site_name", "Moose Pass"), text_field("Notes", ""), checkbox("Fenced", True),
                checkbox("Watered", False)],
            2: [{"/Parent": radio}, {"/Parent": radio}],  # two radio buttons of one field
        },
    )
    pages = read_pdf(data)
    assert len(pages) == 3
    assert pages[0].startswith("Planting Activity Form\nSite name:")
    # Only filled-in fields; an unticked checkbox counts as empty.
    assert pages[0].endswith("Form fields:\nSite_name: Moose Pass\nFenced: Yes")
    assert pages[1] == ""
    assert pages[2].endswith("Form fields:\nSeason: Spring")


def test_pdf_chunks_are_numbered_by_page():
    long_page = "\n".join(f"Row {i}: Picea glauca planted along the north fence" for i in range(30))
    pages = ["Cover page", "", long_page, "Last page"]
    chunks = Chunker(max_chars=400).chunk_pages(pages)

    assert all(c.kind == "pdf" for c in chunks)
    assert chunks[0].text == "--- page 1 ---\nCover page"
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 1)
    long = [c for c in chunks if c.start_line == 3]
    assert len(long) > 1 and all(c.end_line == 3 and len(c.text) <= 400 for c in long)
    assert long[0].text.startswith("--- page 3 ---")
    assert not any(c.start_line == 2 for c in chunks)  # the blank page
    assert chunks[-1].text.endswith("--- page 4 ---\nLast page")

    small = Chunker(max_chars=400).chunk_pages(["one", "two", "three"])
    assert len(small) == 1 and (small[0].start_line, small[0].end_line) == (1, 3)


def test_unreadable_pdfs_say_why():
    with pytest.raises(PdfError, match="OCR"):
        read_pdf(make_pdf(["", ""]))  # what a scan looks like: pages with no text
    with pytest.raises(PdfError, match="password protected"):
        read_pdf(make_pdf(["Secret plans"], password="hunter2"))
    with pytest.raises(PdfError, match="couldn't open"):
        read_pdf(b"%PDF-1.7\nthis is not really a PDF")
    # Encrypted with an empty password (copy protection only) still reads.
    assert read_pdf(make_pdf(["Open to all"], password=""))[0] == "Open to all"


def test_pdfs_are_not_skipped_as_binary(tmp_path):
    for name in ("scan.pdf", "blob.bin"):
        (tmp_path / name).write_bytes(b"%PDF-1.7\n\x00\x01\x02 binary streams")
    pdf = tmp_path / "scan.pdf"
    blob = tmp_path / "blob.bin"
    assert _read(SourceFile(pdf, "scan.pdf", pdf.stat().st_size, 0), 10, 1000)[1] is None
    assert _read(SourceFile(blob, "blob.bin", blob.stat().st_size, 0), 1000, 1000)[1] == "binary"
    # PDFs have their own size limit.
    assert _read(SourceFile(pdf, "scan.pdf", 5000, 0), 1_000_000, 2000)[1] == "larger than 2 KB"


def test_learn_indexes_pdfs_and_cites_their_pages(tmp_path, embedder):
    project = tmp_path / "project"
    (project / "forms").mkdir(parents=True)
    (project / "forms" / "moose.pdf").write_bytes(make_pdf(
        ["Planting Activity Form", "Species planted\nPinus albicaulis whitebark pine seedlings"],
        fields={0: [text_field("Site_name", "Moose Pass")]},
    ))
    (project / "forms" / "scan.pdf").write_bytes(make_pdf([""]))
    big = make_pdf(["x" * 50] * 40)
    (project / "big.pdf").write_bytes(big)

    with Store(tmp_path / "index.db") as store:
        report = learn(project, store, embedder, max_file_bytes=100, max_pdf_bytes=len(big) - 1)
        files = store.files()
        assert files["forms/moose.pdf"].status == "indexed"
        assert files["forms/scan.pdf"].status.startswith("skipped: no text")
        assert files["big.pdf"].status.startswith("skipped: larger than")
        assert report.added == 1 and report.skipped == 2

        hits = search(store, embedder, "whitebark pine seedlings", k=3)
        assert hits[0].chunk.path == "forms/moose.pdf"
        assert hits[0].chunk.kind == "pdf"
        assert format_hit(hits[0]).startswith("### forms/moose.pdf:1-2 (pages)")
        assert "Site_name: Moose Pass" in hits[0].chunk.text


def test_pdfs_learned_before_pdfs_could_be_read_are_read_again_once(tmp_path, embedder):
    project = tmp_path / "project"
    project.mkdir()
    (project / "report.pdf").write_bytes(make_pdf(["Annual planting report"]))
    (project / "notes.md").write_text("# Notes\n")

    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder)
        # What an index from before PDF support holds: the PDF skipped as binary.
        old = store.files()["report.pdf"]
        store.put_file(FileRecord(old.path, old.size, old.mtime, old.sha1, "skipped: binary"))
        store.set_meta("pdf_reader", "0")

        report = learn(project, store, embedder)
        assert report.updated == 1 and report.unchanged == 1  # only the PDF
        assert store.files()["report.pdf"].status == "indexed"

        report = learn(project, store, embedder)
        assert not report.changed and report.unchanged == 2


def test_the_app_shows_a_pdf_and_highlights_the_cited_page(tmp_path, embedder, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from antaigent.config import Settings
    from antaigent.gui.render import file_link
    from antaigent.gui.window import MainWindow
    from antaigent.projects import Registry, index_path

    _app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("antaigent.gui.panels.warm_up", lambda settings: None)
    home, project = tmp_path / "home", tmp_path / "project"
    project.mkdir()
    (project / "report.pdf").write_bytes(make_pdf(["Cover page", "Planting results\nAll seedlings survived"]))
    with Store(index_path(home, project)) as store:
        learn(project, store, embedder)
    Registry(home).touch(project)

    window = MainWindow(Settings(home=home, ollama_url="http://127.0.0.1:9"), project)
    try:
        window.follow_link(file_link("report.pdf", 2, 2))
        shown = window.file.code.toPlainText().split("\n")
        assert shown[0] == "--- page 1 ---" and "text read from the PDF" in window.file.subtitle.text()
        marked = [s.cursor.block().text() for s in window.file.code.extraSelections()]
        assert marked == ["--- page 2 ---", "Planting results", "All seedlings survived"]
        window.follow_link(file_link("report.pdf", 1, 1))  # same file: only the mark moves
        assert [s.cursor.block().text() for s in window.file.code.extraSelections()] == ["--- page 1 ---", "Cover page"]
    finally:
        window.close()


def test_page_citations_highlight_their_pages():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from antaigent.gui.panels import page_lines

    text = "--- page 1 ---\nCover\n\n--- page 3 ---\nThird\nmore\n\n--- page 4 ---\nFourth"
    assert page_lines(text, (1, 1)) == (1, 2)
    assert page_lines(text, (3, 3)) == (4, 6)
    assert page_lines(text, (3, 4)) == (4, 9)
    assert page_lines(text, (2, 2)) == (4, 4)  # page 2 had no text: show where page 3 starts
    assert page_lines(text, (9, 9)) is None
    assert page_lines(text, None) is None
