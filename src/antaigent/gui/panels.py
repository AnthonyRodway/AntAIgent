"""The window's panels: Ask, Search, File, Symbol, Overview, and the sidebar."""

from __future__ import annotations

import datetime as dt
import posixpath
import sqlite3
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..chats import SavedTurn, make_title
from ..index.summarize import file_text
from ..query.context import CODE_KINDS
from ..query.retrieval import Hit
from .render import esc, file_link, link, markdown_html, muted, symbol_link
from .style import GLYPH_ADD, GLYPH_SEARCH, GLYPH_SEND, GLYPH_STOP
from .widgets import Browser, CodeView
from .workers import Task, ask_job, search_job

if TYPE_CHECKING:
    from .window import MainWindow

MAX_FILE_BYTES = 3_000_000
EXAMPLE_QUESTIONS = (
    "What does this project do, and how is it organized?",
    "Where does execution start, and what happens first?",
    "Which parts would I change to add a new feature?",
)


def _vbox(widget: QWidget, margins: int = 0) -> QVBoxLayout:
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(margins, margins, margins, margins)
    return layout


def _page(theme, body: str) -> str:
    return f'<div style="color: {theme.text};">{body}</div>'


def _field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")  # muted, via the stylesheet
    return label


def _search_box(main: MainWindow, placeholder: str) -> QLineEdit:
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setClearButtonEnabled(True)
    box.addAction(main.icon(GLYPH_SEARCH, color=main.theme.muted), QLineEdit.ActionPosition.LeadingPosition)
    return box


def _heading(text: str, level: int = 3) -> str:
    return f"<h{level} style='margin-top: 14px; margin-bottom: 6px;'>{esc(text)}</h{level}>"


def _when(timestamp: float) -> str:
    moment = dt.datetime.fromtimestamp(timestamp)
    today = dt.date.today()
    if moment.date() == today:
        return moment.strftime("%H:%M")
    if moment.date() == today - dt.timedelta(days=1):
        return "Yesterday"
    return moment.strftime("%b %d") if moment.year == today.year else moment.strftime("%Y-%m-%d")


# --- Ask -----------------------------------------------------------------------------


@dataclass
class Turn(SavedTurn):
    done: bool = False
    html: str = ""

    @classmethod
    def restore(cls, saved: SavedTurn) -> Turn:
        return cls(**{f.name: getattr(saved, f.name) for f in fields(SavedTurn)}, done=True)


class AskPanel(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.turns: list[Turn] = []
        self.conversation_id: int | None = None  # None until the first answer is saved
        self.task: Task | None = None
        self._dirty = False

        self.view = Browser()
        self.view.link.connect(self._on_link)
        self.input = QPlainTextEdit()
        self.input.setObjectName("askInput")
        self.input.setPlaceholderText("Ask about the project...   (Ctrl+Enter to send)")
        self.input.setFixedHeight(self.input.fontMetrics().lineSpacing() * 8 + 14)
        self.send = QPushButton(main.icon(GLYPH_SEND, color="#ffffff"), "Ask")
        self.send.setObjectName("primary")
        self.send.setDefault(True)
        self.send.clicked.connect(self.submit)
        self.stop = QPushButton(main.icon(GLYPH_STOP), "Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self._stop)
        self.think = QComboBox()
        self.think.addItems(["low", "medium", "high"])
        self.think.setCurrentText("medium")
        self.think.setToolTip("Reasoning effort. Higher is slower but better for hard questions.")
        self.reasoning = QComboBox()
        self.reasoning.addItem("show", True)
        self.reasoning.addItem("hide", False)
        self.reasoning.setToolTip("Show the model's reasoning above each answer.")
        self.reasoning.currentIndexChanged.connect(lambda _: self._rerender_all())
        for combo in (self.think, self.reasoning):
            combo.setMinimumHeight(28)

        settings = QFormLayout()
        settings.setContentsMargins(0, 4, 0, 0)
        settings.setVerticalSpacing(6)
        settings.addRow(_field_label("Effort"), self.think)
        settings.addRow(_field_label("Reasoning"), self.reasoning)
        controls = QVBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self.send)
        controls.addWidget(self.stop)
        controls.addLayout(settings)
        controls.addStretch(1)
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(self.input, 1)
        bottom.addLayout(controls)

        layout = _vbox(self, 8)
        layout.setSpacing(8)
        layout.addWidget(self.view, 1)
        layout.addLayout(bottom)

        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(keys), self.input, activated=self.submit)
        self._timer = QTimer(self, interval=120)
        self._timer.timeout.connect(self._render_if_dirty)
        self.render()

    # Conversation state

    def reset(self) -> bool:
        """Start a new, empty chat. Returns False while an answer is still streaming."""
        if self.task is not None:
            return False
        self.turns.clear()
        self.conversation_id = None
        self.render()
        return True

    def load(self, conversation_id: int) -> bool:
        """Show a saved chat; new questions continue it. Returns False while an answer is streaming."""
        chats = self.main.chats
        if self.task is not None or chats is None:
            return False
        self.turns = [Turn.restore(t) for t in chats.turns(conversation_id)]
        self.conversation_id = conversation_id
        self.render()
        QTimer.singleShot(0, lambda: self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum()))
        return True

    def abandon(self) -> None:
        """Stop a streaming answer without saving it, e.g. when switching projects."""
        task = self.task
        if task is None:
            return
        for signal in (task.event, task.failed, task.finished):
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass
        task.stop()
        task.wait(5000)
        self.task = None
        self._timer.stop()
        self.set_busy(False)

    def _save(self, turn: Turn) -> None:
        chats = self.main.chats
        if chats is None:
            return
        try:
            if self.conversation_id is None:
                self.conversation_id = chats.create(make_title(turn.question))
            chats.add_turn(self.conversation_id, turn)
        except sqlite3.Error as exc:
            self.main.statusBar().showMessage(f"Couldn't save this chat: {exc}", 8000)
            return
        self.main.sidebar.refresh_chats()

    def set_busy(self, busy: bool) -> None:
        self.send.setEnabled(not busy)
        self.stop.setEnabled(busy)

    def submit(self, question: str | None = None) -> None:
        question = (question or self.input.toPlainText()).strip()
        if not question or self.task is not None or self.main.root is None:
            return
        history = [(t.question, t.answer) for t in self.turns if t.done and t.answer and not t.error][-3:]
        self.turns.append(Turn(question))
        self.input.clear()
        task = Task(ask_job(self.main.settings, self.main.root, question, history, self.think.currentText()), self)
        task.event.connect(self._on_event)
        task.failed.connect(self._on_failed)
        task.finished.connect(self._on_finished)
        self.task = task
        self.set_busy(True)
        self._dirty = True
        self._timer.start()
        task.start()

    def _stop(self) -> None:
        if self.task is not None:
            self.task.stop()
            self.turns[-1].stopped = True

    def _on_event(self, kind: str, payload) -> None:
        turn = self.turns[-1]
        if kind == "context":
            turn.sources = [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in payload.hits]
            turn.graph = payload.symbols + payload.files
            turn.summaries = payload.summaries
        elif kind == "thinking":
            turn.thinking += payload
        elif kind == "content":
            turn.answer += payload
        elif kind == "done":
            turn.stats = payload
        self._dirty = True

    def _on_failed(self, message: str) -> None:
        self.turns[-1].error = message
        self._dirty = True

    def _on_finished(self) -> None:
        if self.turns:
            self.turns[-1].done = True
            self._save(self.turns[-1])
        self.task = None
        self._timer.stop()
        self.set_busy(False)
        self.render(stick=True)
        self.input.setFocus()

    def _on_link(self, url: str) -> None:
        if url.startswith("antai:ask?q="):
            from urllib.parse import unquote

            self.submit(unquote(url[len("antai:ask?q="):]))
        else:
            self.main.follow_link(url)

    # Rendering

    def _render_if_dirty(self) -> None:
        if self._dirty:
            self.render(stick=True)

    def _rerender_all(self) -> None:
        for turn in self.turns:
            turn.html = ""
        self.render()

    def render(self, stick: bool = False) -> None:
        self._dirty = False
        theme = self.main.theme
        if not self.turns:
            self.view.setHtml(_page(theme, self._welcome()))
            return
        parts = []
        for turn in self.turns:
            if turn.done and turn.html:
                parts.append(turn.html)
                continue
            html = self._turn_html(turn)
            if turn.done:
                turn.html = html
            parts.append(html)
        self.view.set_html_keep_scroll(_page(theme, "".join(parts)), stick_to_bottom=stick)

    def _welcome(self) -> str:
        theme = self.main.theme
        if self.main.root is None:
            return (
                "<h2>Welcome to AntAIgent</h2>"
                "<p>Learn a folder to get started: click <b>Learn folder...</b> in the toolbar and pick a "
                "project. AntAIgent indexes it on this machine, then you can ask questions about it here.</p>"
            )
        from urllib.parse import quote

        examples = "".join(
            f"<li>{link('antai:ask?q=' + quote(q), q, theme)}</li>" for q in EXAMPLE_QUESTIONS
        )
        return (
            f"<h2>{esc(self.main.root.name)}</h2>"
            f"<p>{muted(esc(str(self.main.root)), theme)}</p>"
            "<p>Ask anything about this project. Answers cite the files they use; click a citation to open it. "
            "Follow-up questions remember the conversation.</p>"
            f"<p>Try:</p><ul>{examples}</ul>"
            f"<p>{muted('Chats are saved; earlier ones are in the Chats tab of the sidebar.', theme)}</p>"
            f"<p>{muted('Tip: run Summarize once so broad questions get better answers.', theme)}</p>"
        )

    def _turn_html(self, turn: Turn) -> str:
        theme = self.main.theme
        question = esc(turn.question).replace("\n", "<br>")
        out = [
            f'<table width="100%" cellspacing="0" cellpadding="10" bgcolor="{theme.bubble}" style="margin-top: 22px;">'
            f"<tr><td>{question}</td></tr></table>"
        ]
        meta = []
        if turn.sources:
            meta.append(f"{len(turn.sources)} excerpts")
        if turn.graph:
            meta.append("graph: " + ", ".join(turn.graph))
        if turn.summaries:
            meta.append(f"{len(turn.summaries)} summaries")
        if meta:
            out.append(f"<p>{muted(esc(' | '.join(meta)), theme)}</p>")
        if turn.thinking and self.reasoning.currentData():
            out.append(
                f'<p style="color: {theme.muted}; font-style: italic;">{esc(turn.thinking).replace(chr(10), "<br>")}</p>'
            )
        if turn.answer:
            out.append(markdown_html(turn.answer, theme, self.main.paths))
        elif not turn.done and not turn.error:
            status = "Thinking..." if turn.sources else "Searching the index..."
            out.append(f"<p>{muted(status, theme)}</p>")
        if turn.error:
            out.append(f'<p style="color: #d1242f;"><b>Error:</b> {esc(turn.error)}</p>')
        if turn.stopped:
            out.append(f"<p>{muted('Stopped.', theme)}</p>")
        if turn.done and turn.sources:
            sources = ", ".join(
                link(file_link(path, start, end), f"{path}:{start}-{end}", theme) for path, start, end in turn.sources
            )
            out.append(f"<p style='margin-top: 8px;'>{muted('Sources: ', theme)}{sources}</p>")
        if turn.stats:
            s = turn.stats
            generated = s.get("eval_count", 0)
            seconds = s.get("eval_duration", 0) / 1e9
            rate = f" at {generated / seconds:.0f} tok/s" if seconds else ""
            line = (
                f"{s.get('prompt_eval_count', 0):,} prompt tokens, {generated:,} generated{rate}, "
                f"{s.get('total_duration', 0) / 1e9:.1f}s"
            )
            out.append(f"<p>{muted(line, theme)}</p>")
        return "".join(out)


# --- Search ----------------------------------------------------------------------------


class SearchPanel(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.hits: list[Hit] = []
        self.task: Task | None = None

        self.query = _search_box(main, "Search the index (meaning and keywords)...")
        self.query.returnPressed.connect(self.run)
        self.button = QPushButton(main.icon(GLYPH_SEARCH, color="#ffffff"), "Search")
        self.button.setObjectName("primary")
        self.button.clicked.connect(self.run)
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self.query, 1)
        top.addWidget(self.button)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Location", "Symbols", "Vector", "Keyword"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for col in (0, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._preview)
        self.table.cellDoubleClicked.connect(self._open)
        self.preview = CodeView()
        self.hint = QLabel("Double-click a result to open the file.")
        self.hint.setEnabled(False)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.preview)
        splitter.setSizes([300, 400])
        layout = _vbox(self, 8)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.hint)

    def run(self) -> None:
        query = self.query.text().strip()
        if not query or self.task is not None or self.main.root is None:
            return
        self.button.setEnabled(False)
        self.task = Task(search_job(self.main.settings, self.main.root, query, 20), self)
        self.task.succeeded.connect(self._show)
        self.task.failed.connect(lambda msg: self.main.show_error("Search failed", msg))
        self.task.finished.connect(self._done)
        self.task.start()

    def _done(self) -> None:
        self.task = None
        self.button.setEnabled(True)

    def _show(self, hits) -> None:
        self.hits = hits or []
        self.table.setRowCount(len(self.hits))
        for row, hit in enumerate(self.hits):
            c = hit.chunk
            values = [str(row + 1), c.location, ", ".join(c.symbols[:4]),
                      str(hit.vector_rank or "-"), str(hit.keyword_rank or "-")]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
        if self.hits:
            self.table.selectRow(0)
        else:
            self.preview.set_code("No matches.")

    def _preview(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        c = self.hits[rows[0].row()].chunk
        lined = c.kind not in ("data", "notebook")
        self.preview.set_code(c.text, c.path if lined else "", first_line=c.start_line if lined else 1)

    def _open(self, row: int, _col: int) -> None:
        c = self.hits[row].chunk
        self.main.open_file(c.path, c.start_line, c.end_line)


# --- File ------------------------------------------------------------------------------


class FilePanel(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.path: str | None = None
        self.title = QLabel("Pick a file in the sidebar, or click a citation.")
        self.title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        font = self.title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self.title.setFont(font)
        self.subtitle = QLabel("")
        self.subtitle.setEnabled(False)
        self.code = CodeView()
        self.info = Browser()
        self.info.link.connect(main.follow_link)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.code)
        splitter.addWidget(self.info)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 320])
        layout = _vbox(self, 8)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(splitter, 1)

    def clear(self) -> None:
        self.path = None
        self.title.setText("Pick a file in the sidebar, or click a citation.")
        self.subtitle.setText("")
        self.code.set_code("")
        self.info.setHtml("")

    def open(self, path: str, start: int | None = None, end: int | None = None) -> None:
        store, root = self.main.store, self.main.root
        if store is None or root is None:
            return
        mark = (start, end or start) if start else None
        if path == self.path:
            self.code.mark_lines(mark)
            return
        record = store.file(path)
        chunks = store.chunks_for_file(path)
        kind = chunks[0].kind if chunks else ""
        notes = []
        if record is not None and record.status != "indexed":
            notes.append(record.status)
        if kind in ("data", "notebook"):
            text = file_text(chunks)
            notes.append("data profile" if kind == "data" else "notebook cells")
            self.code.set_code(text, mark=None)
        else:
            text, note = self._read(root / path, chunks)
            if note:
                notes.append(note)
            self.code.set_code(text, path, mark=mark)
        if record is not None:
            notes.insert(0, f"{record.size / 1000:,.1f} KB, {record.n_chunks} chunks")
            try:
                if (root / path).stat().st_mtime != record.mtime:
                    notes.append("changed since it was learned")
            except OSError:
                notes.append("no longer on disk")
        self.path = path
        self.title.setText(path)
        self.subtitle.setText(" | ".join(notes))
        self.info.setHtml(_page(self.main.theme, self._info(path)))

    @staticmethod
    def _read(full_path, chunks) -> tuple[str, str]:
        try:
            if full_path.stat().st_size > MAX_FILE_BYTES:
                return file_text(chunks), "too large to show in full; showing the indexed text"
            data = full_path.read_bytes()
        except OSError:
            return file_text(chunks), "showing the indexed text"
        if b"\x00" in data[:8192]:
            return file_text(chunks) or "(binary file)", ""
        try:
            return data.decode("utf-8-sig"), ""
        except UnicodeDecodeError:
            return data.decode("cp1252", errors="replace"), ""

    def _info(self, path: str) -> str:
        store, theme = self.main.store, self.main.theme
        out = []
        summary = store.get_summary("file", path)
        out.append(_heading("Summary"))
        out.append(
            markdown_html(summary.text, theme, self.main.paths)
            if summary else f"<p>{muted('None yet. Run Summarize to create one.', theme)}</p>"
        )

        defs = store.definitions_in(path)
        if defs:
            out.append(_heading("Outline"))
            rows = []
            for d in defs:
                depth = d.qualname.count(".")
                indent = "&nbsp;" * 4 * depth
                rows.append(
                    f"{indent}{link(file_link(path, d.start_line, d.end_line), d.name, theme)} "
                    f"{muted(d.kind, theme)} "
                    f"<a href='{esc(symbol_link(d.qualname))}' style='color: {theme.muted}; text-decoration: none;'>[uses]</a>"
                )
            out.append("<br>".join(rows))

        imports = store.imports_of(path)
        if imports:
            resolved = {(i.line, i.module) for i in imports if i.target}
            local = sorted({i.target for i in imports if i.target})
            external = sorted({i.module for i in imports if (i.line, i.module) not in resolved})
            out.append(_heading("Imports"))
            items = [link(file_link(t), t, theme) for t in local] + [muted(esc(m), theme) for m in external]
            out.append("<br>".join(items))

        importers = sorted({(i.path, i.line) for i in store.imported_by(path)})
        out.append(_heading("Imported by"))
        if importers:
            out.append("<br>".join(link(file_link(p, line, line), f"{p}:{line}", theme) for p, line in importers))
        else:
            out.append(f"<p>{muted('No project file imports it.', theme)}</p>")
        return "".join(out)


# --- Symbol ----------------------------------------------------------------------------


class SymbolPanel(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.view = Browser()
        self.view.link.connect(main.follow_link)
        layout = _vbox(self, 4)
        layout.addWidget(self.view)
        self.clear()

    def clear(self) -> None:
        self.view.setHtml(_page(self.main.theme, "<p>Find a symbol in the sidebar's Symbols tab, or click [uses] in a file's outline.</p>"))

    def show_symbol(self, name: str) -> None:
        store, theme = self.main.store, self.main.theme
        if store is None:
            return
        defs = store.find_definitions(name)
        out = [f"<h2>{esc(name)}</h2>"]
        if not defs:
            similar = store.similar_symbols(name)
            out.append("<p>No definition found.</p>")
            if similar:
                out.append("<p>Similar: " + ", ".join(link(symbol_link(s), s, theme) for s in similar) + "</p>")
            self.view.setHtml(_page(theme, "".join(out)))
            return
        out.append(_heading("Defined at"))
        out.append("<table cellspacing='0' cellpadding='3'>")
        for d in defs:
            out.append(
                f"<tr><td>{link(file_link(d.path, d.start_line, d.end_line), d.location, theme)}</td>"
                f"<td>{muted(d.kind, theme)}</td><td>{esc(d.qualname)}</td></tr>"
            )
        out.append("</table>")

        refs = store.references_to(name, limit=500)
        out.append(_heading(f"Called or referenced from {len(refs):,} place(s)" if refs else "No calls found"))
        if refs:
            out.append("<table cellspacing='0' cellpadding='3'>")
            for r in refs:
                caller = link(symbol_link(r.caller), r.caller, theme) if r.caller else muted("module level", theme)
                out.append(
                    f"<tr><td>{link(file_link(r.path, r.line, r.line), r.location, theme)}</td><td>{caller}</td>"
                    f"<td><code style='color: {theme.muted};'>{esc(r.snippet)}</code></td></tr>"
                )
            out.append("</table>")
            if len(defs) > 1:
                out.append(f"<p>{muted('Several definitions share this name, so call sites may belong to any of them.', theme)}</p>")
        else:
            out.append(f"<p>{muted('It may be unused, an entry point, or only called dynamically.', theme)}</p>")

        if len(defs) == 1 and defs[0].kind in CODE_KINDS:
            d = defs[0]
            callees = list(dict.fromkeys(r.name for r in store.calls_within(d.path, d.qualname)))
            if callees:
                out.append(_heading(f"{d.qualname} calls"))
                known = store.defined_names(callees)
                items = [link(symbol_link(c), c, theme) if c in known else muted(esc(c), theme) for c in callees]
                out.append(", ".join(items))
        out.append(f"<p style='margin-top: 14px;'>{muted('Found by static analysis, so dynamic calls are missed.', theme)}</p>")
        self.view.setHtml(_page(theme, "".join(out)))


# --- Overview --------------------------------------------------------------------------


class OverviewPanel(QWidget):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.view = Browser()
        self.view.link.connect(main.follow_link)
        layout = _vbox(self, 4)
        layout.addWidget(self.view)

    def refresh(self) -> None:
        store, root, theme = self.main.store, self.main.root, self.main.theme
        if store is None or root is None:
            self.view.setHtml(_page(theme, "<p>No project selected.</p>"))
            return
        stats = store.stats()
        out = [f"<h2>{esc(root.name)}</h2>", f"<p>{muted(esc(str(root)), theme)}</p>"]
        overview = store.get_summary("project", "")
        if overview:
            out.append(markdown_html(overview.text, theme, self.main.paths))
            when = dt.datetime.fromtimestamp(overview.created_at).strftime("%Y-%m-%d %H:%M")
            out.append(f"<p>{muted(f'Overview written by {esc(overview.model)} on {when}.', theme)}</p>")
        else:
            out.append(
                "<p><b>No overview yet.</b> Click <b>Summarize</b> in the toolbar to have the model summarize each "
                "file and folder and write an overview. It takes a few seconds per file the first time; after that "
                "only changes are redone.</p>"
            )
        last = store.get_meta("last_learn")
        rows = [
            ("Files indexed", f"{stats['files_indexed']:,}"),
            ("Files skipped", f"{stats['files_skipped']:,}"),
            ("Chunks", f"{stats['chunks']:,}"),
            ("Definitions", f"{stats['symbols']:,}"),
            ("References", f"{stats['references']:,}"),
            ("Imports", f"{stats['imports']:,} ({stats['local_imports']:,} to project files)"),
            ("Summaries", f"{stats['file_summaries']:,} files, {stats['dir_summaries']:,} folders"),
        ]
        if last:
            rows.append(("Last learned", dt.datetime.fromtimestamp(float(last)).strftime("%Y-%m-%d %H:%M")))
        out.append(_heading("Index"))
        out.append("<table cellspacing='0' cellpadding='3'>")
        out.extend(f"<tr><td>{muted(k, theme)}</td><td>{v}</td></tr>" for k, v in rows)
        out.append("</table>")

        folders = sorted(
            (s for (_, key), s in store.summaries("dir").items() if "/" not in key), key=lambda s: s.key.lower()
        )
        if folders:
            out.append(_heading("Top-level folders"))
            out.append("<table cellspacing='0' cellpadding='4'>")
            for s in folders:
                out.append(
                    f"<tr><td valign='top'><b>{esc(s.key)}/</b></td><td>{markdown_html(s.text, theme, self.main.paths)}</td></tr>"
                )
            out.append("</table>")

        skipped = store.skipped_files(100)
        if skipped:
            out.append(_heading(f"Skipped files ({stats['files_skipped']:,})"))
            out.append("<br>".join(f"{esc(p)} {muted(esc(reason), theme)}" for p, reason in skipped))
        self.view.setHtml(_page(theme, "".join(out)))


# --- Sidebar ---------------------------------------------------------------------------


class _TreeItem(QTreeWidgetItem):
    """Sorts folders before files, then by name."""

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        def key(item: QTreeWidgetItem) -> tuple[bool, str]:
            kind = (item.data(0, Qt.ItemDataRole.UserRole) or ("", ""))[0]
            return kind == "file", item.text(0).lower()

        return key(self) < key(other)


class Sidebar(QTabWidget):
    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main

        chats = QWidget()
        self.chat_filter = _search_box(main, "Filter chats...")
        self.chat_filter.textChanged.connect(lambda _: self.refresh_chats())
        new_chat = QPushButton(main.icon(GLYPH_ADD), "New chat")
        new_chat.clicked.connect(lambda: self.main.new_chat())
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self.chat_filter, 1)
        top.addWidget(new_chat)
        self.chat_list = QTreeWidget()
        self.chat_list.setUniformRowHeights(True)
        self.chat_list.setHeaderLabels(["Chat", "Updated"])
        self.chat_list.setRootIsDecorated(False)
        self.chat_list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.chat_list.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.chat_list.header().setStretchLastSection(False)
        self.chat_list.itemClicked.connect(self._open_chat)
        self.chat_list.itemActivated.connect(self._open_chat)
        self.chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._chat_menu)
        self.chat_hint = QLabel("Right-click a chat to rename or delete it.")
        self.chat_hint.setEnabled(False)
        layout = _vbox(chats, 8)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(self.chat_list, 1)
        layout.addWidget(self.chat_hint)
        self.addTab(chats, "Chats")

        files = QWidget()
        self.filter = _search_box(main, "Filter files...")
        self.filter.textChanged.connect(lambda _: self._filter_timer.start())
        self.tree = QTreeWidget()
        self.tree.setUniformRowHeights(True)
        self.tree.setHeaderLabels(["Name", "Chunks"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.itemActivated.connect(self._open_item)
        self.tree.itemClicked.connect(self._open_item)
        layout = _vbox(files, 8)
        layout.setSpacing(8)
        layout.addWidget(self.filter)
        layout.addWidget(self.tree, 1)
        self.addTab(files, "Files")

        symbols = QWidget()
        self.symbol_query = _search_box(main, "Find a function, class...")
        self.symbol_query.textChanged.connect(lambda _: self._symbol_timer.start())
        self.symbol_list = QListWidget()
        self.symbol_list.setUniformItemSizes(True)
        self.symbol_list.itemActivated.connect(self._open_symbol)
        self.symbol_list.itemClicked.connect(self._open_symbol)
        layout = _vbox(symbols, 8)
        layout.setSpacing(8)
        layout.addWidget(self.symbol_query)
        layout.addWidget(self.symbol_list, 1)
        self.addTab(symbols, "Symbols")

        self._filter_timer = QTimer(self, singleShot=True, interval=200)
        self._filter_timer.timeout.connect(self.refresh_files)
        self._symbol_timer = QTimer(self, singleShot=True, interval=200)
        self._symbol_timer.timeout.connect(self.refresh_symbols)

    def refresh(self) -> None:
        self.refresh_files()
        self.refresh_symbols()
        self.refresh_chats()

    def refresh_chats(self) -> None:
        self.chat_list.clear()
        chats = self.main.chats
        if chats is None:
            return
        needle = self.chat_filter.text().strip().lower()
        current = self.main.ask.conversation_id
        for conv in chats.conversations():
            if needle and needle not in conv.title.lower():
                continue
            item = QTreeWidgetItem([conv.title, _when(conv.updated_at)])
            item.setData(0, Qt.ItemDataRole.UserRole, conv.id)
            started = dt.datetime.fromtimestamp(conv.created_at).strftime("%Y-%m-%d %H:%M")
            questions = f"{conv.turns} question{'s' if conv.turns != 1 else ''}"
            item.setToolTip(0, f"{conv.title}\n{questions}, started {started}")
            self.chat_list.addTopLevelItem(item)
            if conv.id == current:
                self.chat_list.setCurrentItem(item)

    def _open_chat(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        self.main.open_chat(item.data(0, Qt.ItemDataRole.UserRole))

    def _chat_menu(self, pos: QPoint) -> None:
        item = self.chat_list.itemAt(pos)
        if item is None or self.main.chats is None:
            return
        conversation_id = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        rename = menu.addAction("Rename...")
        delete = menu.addAction("Delete")
        chosen = menu.exec(self.chat_list.viewport().mapToGlobal(pos))
        if chosen is rename:
            title, ok = QInputDialog.getText(self, "Rename chat", "Name:", text=item.text(0))
            if ok and title.strip():
                self.main.chats.rename(conversation_id, make_title(title))
                self.refresh_chats()
        elif chosen is delete:
            answer = QMessageBox.question(self, "Delete chat", f"Delete \"{item.text(0)}\"? This can't be undone.")
            if answer == QMessageBox.StandardButton.Yes:
                self.main.delete_chat(conversation_id)

    def refresh_files(self) -> None:
        self.tree.clear()
        store = self.main.store
        if store is None:
            return
        needle = self.filter.text().strip().lower()
        summaries = store.summaries()
        folders: dict[str, QTreeWidgetItem] = {}

        def folder(path: str) -> QTreeWidgetItem | None:
            if not path:
                return None
            if path not in folders:
                parent = folder(posixpath.dirname(path))
                item = _TreeItem([posixpath.basename(path), ""])
                item.setData(0, Qt.ItemDataRole.UserRole, ("dir", path))
                s = summaries.get(("dir", path))
                item.setToolTip(0, s.text if s else path + "/")
                (parent.addChild(item) if parent else self.tree.addTopLevelItem(item))
                folders[path] = item
            return folders[path]

        records = sorted(store.files().values(), key=lambda r: (r.path.count("/") == 0, r.path.lower()))
        for rec in records:
            if needle and needle not in rec.path.lower():
                continue
            parent = folder(posixpath.dirname(rec.path))
            indexed = rec.status == "indexed"
            item = _TreeItem([posixpath.basename(rec.path), str(rec.n_chunks) if indexed else ""])
            item.setData(0, Qt.ItemDataRole.UserRole, ("file", rec.path))
            s = summaries.get(("file", rec.path))
            item.setToolTip(0, s.text if s else (rec.path if indexed else f"{rec.path}\n{rec.status}"))
            if not indexed:
                item.setDisabled(True)
            (parent.addChild(item) if parent else self.tree.addTopLevelItem(item))
        self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
        if needle:
            self.tree.expandAll()

    def refresh_symbols(self) -> None:
        self.symbol_list.clear()
        store = self.main.store
        query = self.symbol_query.text().strip()
        if store is None or not query:
            return
        for d in store.search_symbols(query, limit=300):
            item = QListWidgetItem(f"{d.qualname}  ({d.kind})")
            item.setToolTip(d.location)
            item.setData(Qt.ItemDataRole.UserRole, d.qualname)
            self.symbol_list.addItem(item)

    def _open_item(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        kind, path = item.data(0, Qt.ItemDataRole.UserRole) or ("", "")
        if kind == "file" and not item.isDisabled():
            self.main.open_file(path)

    def _open_symbol(self, item: QListWidgetItem) -> None:
        self.main.open_symbol(item.data(Qt.ItemDataRole.UserRole))
