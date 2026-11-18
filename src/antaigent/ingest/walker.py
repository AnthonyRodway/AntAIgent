"""Find the files in a project folder that are worth indexing."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

DEFAULT_IGNORES = [
    ".git/", ".hg/", ".svn/", ".antai/",
    "node_modules/", "bower_components/",
    "__pycache__/", ".venv/", "venv/", ".tox/", ".nox/", "*.egg-info/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".ipynb_checkpoints/",
    ".idea/", ".vscode/", "dist/", "build/", "target/", ".next/", ".cache/",
    "*.lock", "package-lock.json", "pnpm-lock.yaml", "*.min.js", "*.min.css", "*.map",
    "Thumbs.db", "desktop.ini", ".DS_Store",
    # Office writes these lock files next to documents that are open.
    "~$*",
    # Esri file geodatabases are folders of binary tables.
    "*.gdb/",
]

SKIP_EXTENSIONS = frozenset(
    """
    .png .jpg .jpeg .gif .bmp .ico .tif .tiff .webp .svg .psd .heic
    .zip .gz .tgz .bz2 .xz .7z .rar .tar .whl .jar .egg
    .exe .dll .so .dylib .o .a .lib .obj .pdb .class .pyc .pyo .pyd .bin
    .mp3 .mp4 .wav .avi .mov .mkv .flac .ogg .m4a
    .ttf .otf .woff .woff2 .eot
    .pdf .doc .docx .xls .ppt .pptx .odt .ods
    .feather .arrow .pkl .pickle .npy .npz .h5 .hdf5 .nc .mat .sav .rds .rdata
    .mdb .accdb
    .shp .shx .dbf .sbn .sbx .cpg .img .ecw .sid .las .laz
    .gdbtable .gdbtablx .gdbindexes .atx .spx .freelist .horizon .mxd .aprx .lyr .sde .esriaddinx
    """.split()
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel: str  # forward-slash path relative to the project root
    size: int
    mtime: float


def _load_spec(directory: Path, name: str) -> pathspec.PathSpec | None:
    ignore_file = directory / name
    if not ignore_file.is_file():
        return None
    try:
        lines = ignore_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def walk(root: Path) -> Iterator[SourceFile]:
    """Yield candidate files, honouring .gitignore and .antaiignore files at any depth."""
    root = root.resolve()
    defaults = pathspec.GitIgnoreSpec.from_lines(DEFAULT_IGNORES)
    specs: list[tuple[str, pathspec.PathSpec]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        rel_dir = directory.relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        for name in (".gitignore", ".antaiignore"):
            spec = _load_spec(directory, name)
            if spec is not None:
                specs.append((rel_dir, spec))
        active = [(base, spec) for base, spec in specs if rel_dir.startswith(base)]

        def ignored(rel: str, is_dir: bool) -> bool:
            probe = rel + "/" if is_dir else rel
            if defaults.match_file(probe):
                return True
            return any(spec.match_file(probe[len(base):]) for base, spec in active)

        dirnames[:] = sorted(d for d in dirnames if not ignored(rel_dir + d, True))
        for name in sorted(filenames):
            rel = rel_dir + name
            if Path(name).suffix.lower() in SKIP_EXTENSIONS or ignored(rel, False):
                continue
            path = directory / name
            try:
                st = path.stat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                yield SourceFile(path, rel, st.st_size, st.st_mtime)
