"""Reusable widgets: a read-only code view with line numbers, and a link-aware browser."""

from __future__ import annotations

from pygments.lexers import TextLexer
from pygments.styles import get_style_by_name
from pygments.token import Token
from PySide6.QtCore import QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QPainter,
    QPalette,
    QPixmap,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import QPlainTextEdit, QTextBrowser, QTextEdit, QWidget

from .render import DARK, LIGHT, Theme, lexer_for

MAX_HIGHLIGHT_CHARS = 600_000


def is_dark() -> bool:
    hints = QGuiApplication.styleHints()
    if hasattr(hints, "colorScheme"):
        scheme = hints.colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def current_theme() -> Theme:
    return DARK if is_dark() else LIGHT


ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")  # Windows 11, Windows 10


def glyph_icon(glyph: str, color: str, disabled_color: str, gap: float = 0.0) -> QIcon:
    """An icon drawn from Windows' icon font, so it matches the system's own icons.
    Empty where that font is missing; buttons then show just their text.

    `gap` adds blank space to the right, as a fraction of the height, so text beside the
    icon doesn't touch it; size the toolbar's icons to match (e.g. 20x16 for 0.25)."""
    families = QFontDatabase.families()
    family = next((f for f in ICON_FONTS if f in families), None)
    icon = QIcon()
    if family is None:
        return icon
    for size in (16, 20, 24, 32, 48):
        for mode, pen in ((QIcon.Mode.Normal, color), (QIcon.Mode.Disabled, disabled_color)):
            pixmap = QPixmap(size + round(size * gap), size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            font = QFont(family)
            font.setPixelSize(size)
            painter.setFont(font)
            painter.setPen(QColor(pen))
            painter.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, glyph)
            painter.end()
            icon.addPixmap(pixmap, mode)
    return icon


def mono_font(size_delta: int = 0) -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    for family in ("Cascadia Mono", "Consolas"):
        if family in QFontDatabase.families():
            font = QFont(family)
            break
    font.setPointSize(max(8, QGuiApplication.font().pointSize() + size_delta))
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


class _PygmentsHighlighter(QSyntaxHighlighter):
    """Colours a document from one Pygments pass over the whole text, so multi-line
    tokens such as docstrings are coloured correctly."""

    def __init__(self, document):
        super().__init__(document)
        self._lines: list[list[tuple[int, int, QTextCharFormat]]] = []
        self._formats: dict[object, QTextCharFormat | None] = {}
        self._style_name = ""

    def prepare(self, text: str, path: str, language: str, style_name: str) -> None:
        self._lines = []
        if style_name != self._style_name:
            self._formats, self._style_name = {}, style_name
        if not text or len(text) > MAX_HIGHLIGHT_CHARS:
            return
        lexer = lexer_for(path, language)
        if isinstance(lexer, TextLexer):
            return
        style = get_style_by_name(style_name)
        lines: list[list[tuple[int, int, QTextCharFormat]]] = [[]]
        column = 0
        for token_type, value in lexer.get_tokens(text):
            fmt = self._format(style, token_type)
            for i, part in enumerate(value.split("\n")):
                if i:
                    lines.append([])
                    column = 0
                if part and fmt is not None:
                    lines[-1].append((column, len(part), fmt))
                column += len(part)
        self._lines = lines

    def _format(self, style, token_type) -> QTextCharFormat | None:
        if token_type not in self._formats:
            spec = style.style_for_token(token_type)
            fmt = None
            if spec["color"] or spec["bold"] or spec["italic"]:
                fmt = QTextCharFormat()
                if spec["color"]:
                    fmt.setForeground(QColor("#" + spec["color"]))
                if spec["bold"]:
                    fmt.setFontWeight(QFont.Weight.Bold)
                if spec["italic"]:
                    fmt.setFontItalic(True)
            self._formats[token_type] = fmt
        return self._formats[token_type]

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt override)
        number = self.currentBlock().blockNumber()
        if number < len(self._lines):
            for start, length, fmt in self._lines[number]:
                self.setFormat(start, length, fmt)


class _Gutter(QWidget):
    def __init__(self, view: CodeView):
        super().__init__(view)
        self._view = view

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._view.gutter_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        self._view.paint_gutter(event)


class CodeView(QPlainTextEdit):
    """Read-only source view with line numbers, syntax colouring and a highlighted range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(mono_font())
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.first_line = 1
        self._theme = current_theme()
        self._gutter = _Gutter(self)
        self._highlighter = _PygmentsHighlighter(self.document())
        self.blockCountChanged.connect(self._update_margins)
        self.updateRequest.connect(self._update_gutter)
        self._update_margins()
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        style = get_style_by_name(theme.pygments_style)
        fg = style.style_for_token(Token.Text)["color"] or style.style_for_token(Token)["color"]
        fg = f"#{fg}" if fg else theme.text
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {theme.code_bg}; color: {fg}; border: 1px solid {theme.border}; "
            f"border-radius: 6px; selection-background-color: {style.highlight_color}; }}"
        )
        self._gutter_bg = QColor(theme.code_bg)

    def gutter_width(self) -> int:
        digits = len(str(self.first_line + max(1, self.blockCount())))
        return 18 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margins(self, *_) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_margins()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, event) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), self._gutter_bg)
        painter.setPen(QColor(self._theme.muted))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(0, top, self._gutter.width() - 8, height, Qt.AlignmentFlag.AlignRight, str(number + self.first_line))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def set_code(
        self, text: str, path: str = "", *, first_line: int = 1, mark: tuple[int, int] | None = None, language: str = ""
    ) -> None:
        self.first_line = first_line
        self._highlighter.prepare(text, path, language, self._theme.pygments_style)
        self.setPlainText(text)
        self._update_margins()
        self.mark_lines(mark)

    def mark_lines(self, span: tuple[int, int] | None) -> None:
        """Highlight lines start..end (file line numbers) and scroll them into view."""
        if not span:
            self.setExtraSelections([])
            QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(0))
            return
        start, end = span
        doc = self.document()
        first = max(0, start - self.first_line)
        last = min(doc.blockCount() - 1, max(first, end - self.first_line))
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(self._theme.line_highlight))
        fmt.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selections = []
        for n in range(first, min(last, first + 2000) + 1):
            sel = QTextEdit.ExtraSelection()
            sel.cursor = QTextCursor(doc.findBlockByNumber(n))
            sel.format = fmt
            selections.append(sel)
        self.setExtraSelections(selections)
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(max(0, first - 3)))


class Browser(QTextBrowser):
    """Rich-text view whose antai: links are handled by the app; web links open in the browser."""

    link = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.document().setDocumentMargin(12)
        self.anchorClicked.connect(self._clicked)

    def _clicked(self, url: QUrl) -> None:
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
        else:
            self.link.emit(url.toString())

    def set_html_keep_scroll(self, html: str, stick_to_bottom: bool = False) -> None:
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 40
        position = bar.value()
        self.setHtml(html)
        bar.setValue(bar.maximum() if (stick_to_bottom and at_bottom) else position)
