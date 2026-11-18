"""Start the desktop app: `antai gui`, or `antai-gui` without a console window."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from ..config import Settings
from .window import MainWindow


def _icon() -> QIcon:
    icon = QIcon()
    for size in (16, 32, 64, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#2f6fdf"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
        font = QFont("Segoe UI")
        font.setPixelSize(int(size * 0.62))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "A")
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def main(project: Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AntAIgent")
    app.setOrganizationName("AntAIgent")
    app.setWindowIcon(_icon())
    window = MainWindow(Settings.from_env(), project.resolve() if project else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
