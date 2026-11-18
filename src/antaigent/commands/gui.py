"""`antai gui`: open the desktop app."""

from ._common import ProjectOption, fail


def gui(path: ProjectOption = None) -> None:
    """Open the desktop app (also available as `antai-gui`)."""
    try:
        from ..gui.app import main as run_app
    except ImportError as exc:
        fail(f'The desktop app needs PySide6 ({exc}). Install it with: uv pip install -e ".[gui]"')
    run_app(project=path)
