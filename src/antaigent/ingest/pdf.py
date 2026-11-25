"""Read the text of PDF files page by page, with the values filled into their form fields.

Only text the PDF stores as text can be read. Scanned pages are pictures, so a scanned
PDF has no text until it is put through OCR.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import re
import time

PDF_EXTENSIONS = frozenset({".pdf"})

# A PDF can be made to take a very long time to read, so reading stops between pages
# once it has taken this long or produced this much text.
MAX_SECONDS = 60
MAX_CHARS = 5_000_000
# Form field names are found by following parent links, which a damaged file could loop.
MAX_FIELD_DEPTH = 32

# pypdf logs every oddity it works around, and without a handler Python prints those.
logging.getLogger("pypdf").setLevel(logging.CRITICAL)

_BLANK_RUNS = re.compile(r"\n{3,}")


class PdfError(Exception):
    """The PDF can't be read: damaged, password protected, or has no text."""


def available() -> bool:
    return importlib.util.find_spec("pypdf") is not None


def read_pdf(data: bytes) -> list[str]:
    """Return the text of each page, followed by the form fields filled in on it."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import DependencyError
    except ImportError as exc:
        raise PdfError("pypdf isn't installed") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except DependencyError as exc:
                raise PdfError("encrypted, and reading it needs the cryptography package") from exc
            if not unlocked:
                raise PdfError("password protected")
        pages = list(reader.pages)
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(f"pypdf couldn't open it: {exc}") from exc

    started = time.monotonic()
    out: list[str] = []
    chars = 0
    seen_fields: set[str] = set()
    for number, page in enumerate(pages, 1):
        if out and (chars > MAX_CHARS or time.monotonic() - started > MAX_SECONDS):
            out[-1] += f"\n\n(Only the first {number - 1} of {len(pages)} pages were read.)"
            break
        try:
            text = _clean(page.extract_text() or "")
        except Exception:
            text = ""  # one unreadable page shouldn't lose the rest
        fields = _page_fields(page, seen_fields)
        if fields:
            text = (text + "\n\n" if text else "") + "Form fields:\n" + "\n".join(fields)
        out.append(text)
        chars += len(text)
    if not any(page.strip() for page in out):
        raise PdfError("no text in it; scanned pages would need OCR")
    return out


def _clean(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\x00", "").splitlines()]
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def _page_fields(page, seen: set[str]) -> list[str]:
    """`name: value` for each filled-in form field on the page that isn't in `seen`.
    A field can have several widgets (radio buttons), so each name is listed once."""
    lines: list[str] = []
    try:
        annotations = page["/Annots"] if "/Annots" in page else []
    except Exception:
        return lines
    for ref in annotations or []:
        try:
            widget = ref.get_object()
            if widget.get("/Subtype") != "/Widget":
                continue
            name, value = _field(widget)
        except Exception:
            continue
        if name and value and name not in seen:
            seen.add(name)
            lines.append(f"{name}: {value}")
    return lines


def _field(widget) -> tuple[str, str]:
    """A widget's full field name (parent names joined with dots) and its value."""
    names: list[str] = []
    value = None
    node = widget
    for _ in range(MAX_FIELD_DEPTH):
        if "/T" in node:
            names.append(str(node["/T"]))
        if value is None and "/V" in node:
            value = node["/V"]
        if "/Parent" not in node:
            break
        node = node["/Parent"]
    return ".".join(reversed(names)), _value(value)


def _value(value) -> str:
    from pypdf.generic import NameObject

    if value is None:
        return ""
    value = value.get_object() if hasattr(value, "get_object") else value
    if isinstance(value, NameObject):  # a checkbox or radio button: /Off, /Yes or a choice's name
        return "" if value == "/Off" else value[1:]
    if isinstance(value, list):  # several choices picked in a list box
        return ", ".join(v for v in (_value(item) for item in value) if v)
    text = value.decode("latin-1") if isinstance(value, bytes) else str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
