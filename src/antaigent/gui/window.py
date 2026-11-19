"""The main window: toolbar, sidebar, panels, status bar, and the background jobs."""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..chats import ChatStore, chats_path
from ..config import Settings
from ..index.indexer import LearnReport
from ..index.store import Store
from ..index.summarize import SummaryReport
from ..llm.ollama import has_model
from ..projects import Registry, index_path
from . import style
from .panels import (
    AskPanel,
    FilePanel,
    OverviewPanel,
    SearchPanel,
    Sidebar,
    SymbolPanel,
)
from .render import parse_link
from .style import GLYPH_ADD, GLYPH_DOCUMENT, GLYPH_FOLDER, GLYPH_REFRESH, GLYPH_WATCH
from .widgets import current_theme, glyph_icon
from .workers import Task, learn_job, status_job, summarize_job, watch_job

OLLAMA_CHECK_MS = 30_000


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, project: Path | None = None):
        super().__init__()
        self.settings = settings
        self.registry = Registry(settings.home)
        self.theme = current_theme()
        style.apply(QApplication.instance(), self.theme)
        self.root: Path | None = None
        self.store: Store | None = None
        self.chats: ChatStore | None = None
        self.paths: set[str] = set()
        self.job: Task | None = None
        self.watcher: Task | None = None
        self._status_task: Task | None = None
        self._update_pending = False
        self.setWindowTitle("AntAIgent")

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setObjectName("main")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(20, 16))  # 16px glyphs plus the gap glyph_icon leaves before the text
        self.addToolBar(toolbar)

        self._folder_icon = self.icon(GLYPH_FOLDER)
        self.projects = QComboBox()
        self.projects.setToolTip("Learned projects")
        self.projects.setMinimumContentsLength(24)
        self.projects.setMinimumHeight(30)
        self.projects.setIconSize(QSize(16, 16))
        self.projects.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.projects.currentIndexChanged.connect(self._project_chosen)
        toolbar.addWidget(self.projects)
        self.act_learn = QAction(self.icon(GLYPH_ADD, gap=0.25), "Learn folder", self, triggered=self.learn_folder)
        self.act_learn.setToolTip("Index a new folder and add it to the project list")
        toolbar.addAction(self.act_learn)
        toolbar.addSeparator()
        self.act_update = QAction(
            self.icon(GLYPH_REFRESH, gap=0.25), "Update", self, triggered=lambda: self.start_learn(self.root)
        )
        self.act_update.setToolTip("Re-learn the current project; only changed files are processed")
        self.act_summarize = QAction(self.icon(GLYPH_DOCUMENT, gap=0.25), "Summarize", self, triggered=self.start_summarize)
        self.act_summarize.setToolTip("Have the model summarize each file and folder and write a project overview")
        self.act_watch = QAction(self.icon(GLYPH_WATCH, gap=0.25), "Watch", self, checkable=True, toggled=self.toggle_watch)
        self.act_watch.setToolTip("Re-learn automatically whenever files in the project change")
        for action in (self.act_update, self.act_summarize, self.act_watch):
            toolbar.addAction(action)

        # Panels
        self.sidebar = Sidebar(self)
        self.tabs = QTabWidget()
        self.ask = AskPanel(self)
        self.search = SearchPanel(self)
        self.file = FilePanel(self)
        self.symbol = SymbolPanel(self)
        self.overview = OverviewPanel(self)
        for panel, title in ((self.ask, "Ask"), (self.search, "Search"), (self.file, "File"),
                             (self.symbol, "Symbol"), (self.overview, "Overview")):
            self.tabs.addTab(panel, title)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([300, 1100])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 2, 10, 0)  # so the cards sit inside the window's edge
        layout.addWidget(self.splitter)
        self.setCentralWidget(central)

        # Status bar
        self.stats_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setTextVisible(False)
        self.job_label = QLabel("")
        self.job_label.setTextFormat(Qt.TextFormat.PlainText)  # shows file paths
        self.job_label.setMaximumWidth(380)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.cancel_job)
        self.ollama_label = QLabel("Checking Ollama...")
        bar = self.statusBar()
        bar.addWidget(self.stats_label, 1)
        for widget in (self.job_label, self.progress, self.cancel, self.ollama_label):
            bar.addPermanentWidget(widget)
        self._show_job(False)

        self._restore_layout()
        self.refresh_projects(select=project or self.registry.resolve(None))
        self._ollama_timer = QTimer(self, interval=OLLAMA_CHECK_MS)
        self._ollama_timer.timeout.connect(self.check_ollama)
        self._ollama_timer.start()
        QTimer.singleShot(0, self.check_ollama)

    # --- Projects ----------------------------------------------------------------

    def refresh_projects(self, select: Path | None = None) -> None:
        current = select or self.root
        self.projects.blockSignals(True)
        self.projects.clear()
        for path, _last_used in self.registry.all():
            if index_path(self.settings.home, path).exists():
                self.projects.addItem(self._folder_icon, path.name, str(path))
                self.projects.setItemData(self.projects.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        index = self.projects.findData(str(current)) if current else -1
        if index < 0 and self.projects.count():
            index = 0
        self.projects.setCurrentIndex(index)
        self.projects.blockSignals(False)
        self._project_chosen(index)

    def _project_chosen(self, index: int) -> None:
        path = Path(self.projects.itemData(index)) if index >= 0 else None
        if path == self.root and self.store is not None:
            return
        if self.watcher is not None:
            self.act_watch.setChecked(False)
        self.open_project(path)

    def open_project(self, root: Path | None) -> None:
        self.ask.abandon()
        if self.store is not None:
            self.store.close()
            self.store = None
        if self.chats is not None:
            self.chats.close()
            self.chats = None
        self.root = root
        if root is not None:
            self.store = Store(index_path(self.settings.home, root))
            self.chats = ChatStore(chats_path(self.settings.home, root))
            self.registry.touch(root)
            self.setWindowTitle(f"{root.name} - AntAIgent")
        else:
            self.setWindowTitle("AntAIgent")
        has_project = root is not None
        for action in (self.act_update, self.act_summarize, self.act_watch):
            action.setEnabled(has_project)
        self.file.clear()
        self.symbol.clear()
        self.ask.reset()
        self.refresh_views()

    def refresh_views(self) -> None:
        """Re-read the index after it changed."""
        self.paths = set(self.store.files()) if self.store is not None else set()
        self.sidebar.refresh()
        self.overview.refresh()
        self.ask.render()
        if self.store is not None:
            s = self.store.stats()
            self.stats_label.setText(
                f"{s['files_indexed']:,} files, {s['chunks']:,} chunks, {s['symbols']:,} definitions, "
                f"{s['file_summaries']:,} summaries"
            )
        else:
            self.stats_label.setText("No project. Use Learn folder... to add one.")
        if self.file.path:
            path, self.file.path = self.file.path, None
            if path in self.paths:
                self.file.open(path)
            else:
                self.file.clear()

    # --- Navigation --------------------------------------------------------------

    def follow_link(self, url: str) -> None:
        kind, params = parse_link(url)
        if kind == "file" and "p" in params:
            start, end = params.get("s", ""), params.get("e", "")
            # Links in answers come from the model, so the line numbers may not be numbers.
            self.open_file(
                params["p"], int(start) if start.isdigit() else None, int(end) if end.isdigit() else None
            )
        elif kind == "symbol" and "n" in params:
            self.open_symbol(params["n"])

    def open_file(self, path: str, start: int | None = None, end: int | None = None) -> None:
        if self.file.open(path, start, end):
            self.tabs.setCurrentWidget(self.file)

    def open_symbol(self, name: str) -> None:
        self.symbol.show_symbol(name)
        self.tabs.setCurrentWidget(self.symbol)

    # --- Chats ---------------------------------------------------------------------

    def new_chat(self) -> None:
        if not self.ask.reset():
            self.statusBar().showMessage("Wait for the answer to finish, or press Stop.", 5000)
            return
        self.sidebar.refresh_chats()
        self.tabs.setCurrentWidget(self.ask)
        self.ask.input.setFocus()

    def open_chat(self, conversation_id: int) -> None:
        if conversation_id != self.ask.conversation_id and not self.ask.load(conversation_id):
            self.statusBar().showMessage("Wait for the answer to finish, or press Stop.", 5000)
            self.sidebar.refresh_chats()
            return
        self.tabs.setCurrentWidget(self.ask)

    def delete_chat(self, conversation_id: int) -> None:
        if self.chats is None:
            return
        if conversation_id == self.ask.conversation_id:
            if not self.ask.reset():
                self.statusBar().showMessage("Wait for the answer to finish before deleting this chat.", 5000)
                return
        self.chats.delete(conversation_id)
        self.sidebar.refresh_chats()

    # --- Jobs --------------------------------------------------------------------

    def _show_job(self, visible: bool, label: str = "") -> None:
        self.job_label.setText(label)
        for widget in (self.job_label, self.progress, self.cancel):
            widget.setVisible(visible)
        self.progress.setRange(0, 0)
        self.cancel.setEnabled(True)
        for action in (self.act_learn, self.act_update, self.act_summarize):
            action.setEnabled(not visible and (action is self.act_learn or self.root is not None))

    def _start_job(self, label: str, task: Task, on_success) -> bool:
        if self.job is not None:
            self.statusBar().showMessage("Wait for the current job to finish first.", 5000)
            return False
        task.setParent(self)
        task.progress.connect(self._on_progress)
        task.succeeded.connect(on_success)
        task.failed.connect(lambda msg: self.show_error(f"{label} failed", msg))
        task.finished.connect(self._job_finished)
        self.job = task
        self._show_job(True, label)
        task.start()
        return True

    def _on_progress(self, done: int, total: int, current: str) -> None:
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        if not current:
            return
        metrics = self.job_label.fontMetrics()
        count = f"{done:,} / {total:,}  " if total else ""
        self.job_label.setText(count + metrics.elidedText(current, Qt.TextElideMode.ElideMiddle, 320))

    def _job_finished(self) -> None:
        self.job = None
        self._show_job(False)
        if self._update_pending and self.root is not None:
            self._update_pending = False
            self.start_learn(self.root, quiet=True)

    def cancel_job(self) -> None:
        if self.job is not None:
            self.job.stop()
            self.cancel.setEnabled(False)
            self.job_label.setText("Stopping...")

    def learn_folder(self) -> None:
        start = str(self.root.parent) if self.root else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choose a folder to learn", start)
        if chosen:
            self.start_learn(Path(chosen).resolve())

    def start_learn(self, root: Path | None, quiet: bool = False) -> None:
        if root is None:
            return

        def done(report: LearnReport | None) -> None:
            if report is None:
                self.statusBar().showMessage("Learning stopped. Files finished so far are saved.", 8000)
            elif not quiet or report.changed:
                parts = [f"{report.added:,} new", f"{report.updated:,} updated", f"{report.removed:,} removed"]
                self.statusBar().showMessage(
                    f"Learned {root.name} in {report.seconds:.1f}s: {', '.join(parts)}.", 10000
                )
            if root != self.root:
                self.refresh_projects(select=root)
            elif report is None or report.changed:
                self.refresh_views()

        self._start_job(f"Learning {root.name}", Task(learn_job(self.settings, root)), done)

    def start_summarize(self) -> None:
        if self.root is None:
            return

        def done(report: SummaryReport | None) -> None:
            if report is not None:
                text = f"Summarized {report.files:,} files and {report.dirs:,} folders in {report.seconds:.0f}s"
                if report.project:
                    text += ", and wrote the project overview"
                if report.stopped:
                    text += " (stopped early)"
                self.statusBar().showMessage(text + ".", 12000)
                if report.errors:
                    self.show_error("Some summaries failed", "\n".join(report.errors[:8]))
            self.refresh_views()

        self._start_job(f"Summarizing {self.root.name}", Task(summarize_job(self.settings, self.root)), done)

    def toggle_watch(self, on: bool) -> None:
        if on and self.root is not None and self.watcher is None:
            task = Task(watch_job(self.root), self)
            task.event.connect(self._files_changed)
            task.failed.connect(lambda msg: (self.show_error("Watching stopped", msg), self.act_watch.setChecked(False)))
            task.finished.connect(self._watch_finished)
            self.watcher = task
            self.act_watch.setText("Watching")
            task.start()
            self.statusBar().showMessage(f"Watching {self.root.name} for changes.", 5000)
        elif not on and self.watcher is not None:
            self.watcher.stop()

    def _watch_finished(self) -> None:
        self.watcher = None
        self.act_watch.setText("Watch")
        if self.act_watch.isChecked():
            self.act_watch.setChecked(False)

    def _files_changed(self, kind: str, _paths) -> None:
        if kind != "changed" or self.root is None:
            return
        if self.job is not None:
            self._update_pending = True
        else:
            self.start_learn(self.root, quiet=True)

    # --- Ollama status -------------------------------------------------------------

    def check_ollama(self) -> None:
        if self._status_task is not None:
            return
        task = Task(status_job(self.settings), self)
        task.succeeded.connect(self._show_ollama)
        task.failed.connect(
            lambda msg: self.ollama_label.setText(f"<span style='color:#d1242f'>Ollama: {html.escape(msg)}</span>")
        )
        task.finished.connect(lambda: setattr(self, "_status_task", None))
        self._status_task = task
        task.start()

    def _show_ollama(self, info: dict) -> None:
        missing = [m for m in (self.settings.chat_model, self.settings.embed_model) if not has_model(info["models"], m)]
        if missing:
            self.ollama_label.setText(f"<span style='color:#d1242f'>Missing models: {html.escape(', '.join(missing))}</span>")
            self.ollama_label.setToolTip("Download with: " + "; ".join(f"ollama pull {m}" for m in missing))
            return
        chat = next((m for m in info["running"] if m.get("name") == self.settings.chat_model), None)
        if chat:
            size = chat.get("size") or 0
            gpu = 100 * (chat.get("size_vram") or 0) / size if size else 0
            state = f"{self.settings.chat_model} loaded, {gpu:.0f}% GPU"
        else:
            state = f"{self.settings.chat_model} ready (loads on first question)"
        self.ollama_label.setText(f"Ollama {info['version']} | {state}")
        self.ollama_label.setToolTip("")

    # --- Misc ----------------------------------------------------------------------

    def icon(self, glyph: str, color: str | None = None, gap: float = 0.0) -> QIcon:
        """A Fluent icon in the theme's text colour (or `color`). Toolbar icons use gap=0.25."""
        return glyph_icon(glyph, color or self.theme.text, self.theme.muted, gap=gap)

    def show_error(self, title: str, message: str) -> None:
        box = QMessageBox(QMessageBox.Icon.Warning, title, message, QMessageBox.StandardButton.Ok, self)
        box.setTextFormat(Qt.TextFormat.PlainText)  # error text can include file names and paths
        box.exec()

    def _restore_layout(self) -> None:
        settings = QSettings("AntAIgent", "AntAIgent")
        geometry = settings.value("geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        else:
            self.resize(1400, 900)
        splitter = settings.value("splitter")
        if isinstance(splitter, QByteArray):
            self.splitter.restoreState(splitter)
        settings.remove("think")  # replaced by "effort", so the old saved default doesn't override the new one
        effort = settings.value("effort")
        if effort in ("low", "medium", "high"):
            self.ask.think.setCurrentText(effort)
        reasoning = settings.value("reasoning")
        if reasoning in ("show", "hide"):
            self.ask.reasoning.setCurrentText(reasoning)

    def closeEvent(self, event) -> None:  # noqa: N802
        settings = QSettings("AntAIgent", "AntAIgent")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("splitter", self.splitter.saveState())
        settings.setValue("effort", self.ask.think.currentText())
        settings.setValue("reasoning", self.ask.reasoning.currentText())
        for task in (self.job, self.watcher, self.ask.task, self.search.task, self._status_task):
            if task is not None:
                task.stop()
        for task in (self.job, self.watcher, self.ask.task, self.search.task, self._status_task):
            if task is not None:
                task.wait(5000)
        if self.store is not None:
            self.store.close()
        if self.chats is not None:
            self.chats.close()
        super().closeEvent(event)
