"""The app's look: a palette and one stylesheet, both derived from the light or dark theme.

Content sits on rounded cards over a slightly darker window, in the spirit of Windows 11.
Combo boxes are left to the native style and pick up the palette.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .render import Theme

# Code points in the Segoe Fluent Icons / Segoe MDL2 Assets fonts.
GLYPH_FOLDER = chr(0xE8B7)
GLYPH_ADD = chr(0xE710)
GLYPH_REFRESH = chr(0xE72C)
GLYPH_DOCUMENT = chr(0xE8A5)
GLYPH_WATCH = chr(0xE890)
GLYPH_SEND = chr(0xE724)
GLYPH_STOP = chr(0xE71A)
GLYPH_SEARCH = chr(0xE721)


def rgba(hex_color: str, alpha: float) -> str:
    color = QColor(hex_color)
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


def blend(top: str, bottom: str, alpha: float) -> str:
    """`top` at `alpha` over `bottom`, as a solid colour. Item views paint selection in several
    layers, so a translucent colour would come out darker than intended."""
    a, b = QColor(top), QColor(bottom)
    mix = [round(x * alpha + y * (1 - alpha)) for x, y in ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue()))]
    return QColor(*mix).name()


def palette(theme: Theme) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: theme.window,
        QPalette.ColorRole.WindowText: theme.text,
        QPalette.ColorRole.Base: theme.surface,
        QPalette.ColorRole.AlternateBase: theme.raised,
        QPalette.ColorRole.Text: theme.text,
        QPalette.ColorRole.Button: theme.raised,
        QPalette.ColorRole.ButtonText: theme.text,
        QPalette.ColorRole.Highlight: theme.accent_fill,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: theme.muted,
        QPalette.ColorRole.ToolTipBase: theme.raised,
        QPalette.ColorRole.ToolTipText: theme.text,
        QPalette.ColorRole.Link: theme.accent,
        QPalette.ColorRole.Light: theme.raised,
        QPalette.ColorRole.Midlight: theme.border,
        QPalette.ColorRole.Mid: theme.border,
        QPalette.ColorRole.Dark: theme.window,
    }
    if hasattr(QPalette.ColorRole, "Accent"):  # Qt 6.6+
        roles[QPalette.ColorRole.Accent] = theme.accent_fill
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(theme.muted))
    return p


def stylesheet(theme: Theme) -> str:
    t = theme
    hover = rgba("#ffffff", 0.06) if t.dark else rgba("#000000", 0.05)
    pressed = rgba("#ffffff", 0.10) if t.dark else rgba("#000000", 0.09)
    selected = rgba(t.accent, 0.24 if t.dark else 0.13)
    # Solid versions for item views, which layer translucent colours.
    item_hover = blend("#ffffff", t.surface, 0.06) if t.dark else blend("#000000", t.surface, 0.05)
    item_selected = blend(t.accent, t.surface, 0.24 if t.dark else 0.13)
    scroll = rgba("#ffffff", 0.22) if t.dark else rgba("#000000", 0.25)
    fill_hover = QColor(t.accent_fill).lighter(112).name()
    return f"""
    QMainWindow, QDialog {{ background: {t.window}; }}
    QWidget {{ color: {t.text}; }}
    QLabel:disabled, QLabel#fieldLabel {{ color: {t.muted}; }}
    QToolTip {{ background: {t.raised}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 6px; }}

    /* Toolbar */
    QToolBar#main {{ background: {t.window}; spacing: 4px; padding: 6px 10px; border: none; }}
    QToolBar#main::separator {{ width: 1px; background: {t.border}; margin: 6px 8px; }}
    QToolBar#main QToolButton {{ padding: 5px 10px; border: 1px solid transparent; border-radius: 6px; }}
    QToolBar#main QToolButton:hover {{ background: {hover}; }}
    QToolBar#main QToolButton:pressed {{ background: {pressed}; }}
    QToolBar#main QToolButton:checked {{ background: {rgba(t.accent, 0.22)}; border-color: {t.accent}; }}
    QToolBar#main QToolButton:disabled {{ color: {t.muted}; }}

    /* Cards and pill tabs */
    QTabWidget::pane {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; top: 0px; }}
    QTabWidget::tab-bar {{ alignment: left; left: 2px; }}
    QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: transparent; color: {t.muted}; border: none; border-radius: 6px;
        padding: 5px 12px; margin: 0px 2px 6px 0px;
    }}
    QTabBar::tab:hover {{ background: {hover}; color: {t.text}; }}
    QTabBar::tab:selected {{ background: {selected}; color: {t.text}; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:horizontal {{ width: 8px; }}
    QSplitter::handle:vertical {{ height: 8px; }}

    /* Lists, trees and tables */
    QTreeView, QListView, QTableView, QTextBrowser {{
        background: {t.surface}; border: none; outline: 0;
        selection-background-color: {item_selected}; selection-color: {t.text};
    }}
    QTreeView::item, QListView::item {{ padding: 3px 4px; border-radius: 4px; }}
    QTreeView::item:hover, QListView::item:hover, QTableView::item:hover {{ background: {item_hover}; }}
    QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
        background: {item_selected}; color: {t.text};
    }}
    QTreeView::item:disabled {{ color: {t.muted}; }}
    QTableView {{ gridline-color: {t.border}; }}
    QTableView::item {{ padding: 0px 6px; }}
    QHeaderView {{ background: {t.surface}; border: none; }}
    QHeaderView::section {{
        background: {t.surface}; color: {t.muted}; border: none; border-bottom: 1px solid {t.border};
        padding: 5px 8px;
    }}
    QTableCornerButton::section {{ background: {t.surface}; border: none; }}

    /* Inputs */
    QLineEdit, QPlainTextEdit#askInput {{
        background: {t.raised}; border: 1px solid {t.border}; border-radius: 6px; padding: 5px 8px;
        selection-background-color: {t.accent_fill}; selection-color: #ffffff;
    }}
    QLineEdit:hover, QPlainTextEdit#askInput:hover {{ border-color: {QColor(t.border).lighter(125).name() if t.dark else QColor(t.border).darker(115).name()}; }}
    QLineEdit:focus, QPlainTextEdit#askInput:focus {{ border: 1px solid {t.accent}; }}

    /* Buttons */
    QPushButton {{
        background: {t.raised}; color: {t.text}; border: 1px solid {t.border}; border-radius: 6px;
        padding: 5px 14px; min-height: 20px;
    }}
    QPushButton:hover {{ background: {QColor(t.raised).lighter(112).name() if t.dark else QColor(t.raised).darker(104).name()}; }}
    QPushButton:pressed {{ background: {pressed}; }}
    QPushButton:disabled {{ color: {t.muted}; background: transparent; }}
    QPushButton#primary {{ background: {t.accent_fill}; border-color: {t.accent_fill}; color: #ffffff; }}
    QPushButton#primary:hover {{ background: {fill_hover}; border-color: {fill_hover}; }}
    QPushButton#primary:disabled {{ background: {rgba(t.accent_fill, 0.35)}; border-color: transparent; color: {rgba("#ffffff", 0.6)}; }}

    /* Menus */
    QMenu {{ background: {t.raised}; border: 1px solid {t.border}; border-radius: 8px; padding: 4px; }}
    QMenu::item {{ padding: 6px 20px 6px 12px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {hover}; }}
    QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 6px; }}

    /* Status bar */
    QStatusBar {{ background: {t.window}; color: {t.muted}; }}
    QStatusBar::item {{ border: none; }}
    QStatusBar QLabel {{ color: {t.muted}; padding: 0px 6px; }}
    QProgressBar {{ background: {t.raised}; border: none; border-radius: 3px; max-height: 6px; min-width: 160px; }}
    QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}

    /* Slim scrollbars */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {scroll}; border-radius: 3px; }}
    QScrollBar::handle:hover {{ background: {t.muted}; }}
    QScrollBar::handle:vertical {{ min-height: 30px; }}
    QScrollBar::handle:horizontal {{ min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    """


def apply(app: QApplication, theme: Theme) -> None:
    app.setPalette(palette(theme))
    app.setStyleSheet(stylesheet(theme))
