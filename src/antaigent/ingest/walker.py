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
    # Secrets: the index keeps a copy of everything it reads, and answers can quote it.
    ".env", ".env.*", "!.env.example", "!.env.sample", "!.env.template",
    "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    ".netrc", ".pypirc", ".npmrc",
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


def _load_spec(path: str) -> pathspec.PathSpec | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _suffix(name: str) -> str:
    """Lower-cased extension, by the same rule as Path.suffix."""
    i = name.rfind(".")
    return name[i:].lower() if 0 < i < len(name) - 1 else ""


def _is_junction(entry: os.DirEntry) -> bool:
    if hasattr(entry, "is_junction"):  # Python 3.12+
        return entry.is_junction()
    try:  # Windows on older Pythons: a junction is a mount-point reparse point
        tag = entry.stat(follow_symlinks=False).st_reparse_tag
    except (OSError, AttributeError):
        return False
    return tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)


def _leaves(path: str, root: Path) -> bool:
    """Whether a symlink or junction points outside the project."""
    try:
        return not Path(path).resolve().is_relative_to(root)
    except OSError:
        return True


def walk(root: Path) -> Iterator[SourceFile]:
    """Yield candidate files, honouring .gitignore and .antaiignore files at any depth.

    Folders are visited depth first in name order, each folder's files before its
    subfolders. Symlinked folders aren't entered, and other links that lead outside the
    folder (file symlinks, junctions) are skipped, so learning a folder never reads files
    from elsewhere on the machine.

    os.scandir already knows each entry's type and link status, so the only system
    call per file is the stat for its size and time."""
    root = root.resolve()
    defaults = pathspec.GitIgnoreSpec.from_lines(DEFAULT_IGNORES)
    specs: list[tuple[str, pathspec.PathSpec]] = []
    stack: list[tuple[str, str]] = [(str(root), "")]  # (folder, its path relative to root with a trailing /)

    while stack:
        directory, rel_dir = stack.pop()
        try:
            with os.scandir(directory) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name in (".gitignore", ".antaiignore") and entry.is_file():
                spec = _load_spec(entry.path)
                if spec is not None:
                    specs.append((rel_dir, spec))
        active = [(base, spec) for base, spec in specs if rel_dir.startswith(base)]

        def ignored(rel: str, is_dir: bool) -> bool:
            probe = rel + "/" if is_dir else rel
            if defaults.match_file(probe):
                return True
            return any(spec.match_file(probe[len(base):]) for base, spec in active)

        subdirs: list[tuple[str, str]] = []
        for entry in entries:
            name, rel = entry.name, rel_dir + entry.name
            try:
                is_dir = entry.is_dir()  # follows links, like os.walk
            except OSError:
                is_dir = False
            if is_dir:
                if ignored(rel, True) or entry.is_symlink():
                    continue
                if _is_junction(entry) and _leaves(entry.path, root):
                    continue
                subdirs.append((entry.path, rel + "/"))
                continue
            if _suffix(name) in SKIP_EXTENSIONS or ignored(rel, False):
                continue
            try:
                # A real stat rather than the cached scandir one: Windows can update a
                # folder's listing late for a file still open for writing.
                st = os.stat(entry.path)
            except OSError:
                continue
            if entry.is_symlink() and _leaves(entry.path, root):
                continue
            if stat.S_ISREG(st.st_mode):
                yield SourceFile(Path(entry.path), rel, st.st_size, st.st_mtime)
        stack.extend(reversed(subdirs))
