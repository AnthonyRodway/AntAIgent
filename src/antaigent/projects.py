"""Where each project's index lives, and which project a command refers to.

Indexes are kept under the AntAIgent home folder (~/.antai by default) rather than
inside the learned folder, so learning never writes into your code or data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


def _norm(root: Path) -> str:
    return os.path.normcase(str(root.resolve()))


def index_dir(home: Path, root: Path) -> Path:
    root = root.resolve()
    digest = hashlib.sha1(_norm(root).encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", root.name).strip("-")[:40] or "root"
    return home / "projects" / f"{slug}-{digest}"


def index_path(home: Path, root: Path) -> Path:
    return index_dir(home, root) / "index.db"


class Registry:
    """Remembers learned projects so `ask` works without repeating the folder path."""

    def __init__(self, home: Path):
        self.path = home / "projects.json"

    def _load(self) -> dict[str, dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def touch(self, root: Path) -> None:
        data = self._load()
        data[_norm(root)] = {"root": str(root.resolve()), "last_used": time.time()}
        self._save(data)

    def all(self) -> list[tuple[Path, float]]:
        entries = [(Path(v["root"]), float(v.get("last_used", 0))) for v in self._load().values()]
        return sorted(entries, key=lambda e: e[1], reverse=True)

    def resolve(self, path: Path | None) -> Path | None:
        """An explicit path wins; otherwise the learned project containing the
        current folder; otherwise the most recently used project."""
        if path is not None:
            return path.resolve()
        known = {_norm(root): root for root, _ in self.all()}
        cwd = Path.cwd().resolve()
        for candidate in (cwd, *cwd.parents):
            if _norm(candidate) in known:
                return known[_norm(candidate)]
        recent = self.all()
        return recent[0][0] if recent else None
