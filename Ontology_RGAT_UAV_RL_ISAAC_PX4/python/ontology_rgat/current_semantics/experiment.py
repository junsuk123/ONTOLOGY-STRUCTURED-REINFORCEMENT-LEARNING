# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Run bookkeeping of the experiments: results folder, single-writer lock, atomic writes."""
from __future__ import annotations

import contextlib
import datetime as dt
import errno
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterator

import numpy as np

from . import SCHEMA_ID
from .contracts import PROJECT_ROOT, CurrentSemanticsConfig, contract_manifest, load_config

REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "current_semantics_3d" / SCHEMA_ID
LOCK_NAME = "run.lock"


def run_root_for(config: CurrentSemanticsConfig, requested: Path | None) -> Path:
    """``requested``, or the default results folder."""
    return DEFAULT_RUN_ROOT if requested is None else requested


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), *args], check=True,
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def provenance() -> dict[str, Any]:
    """Source and runtime identity of this invocation (read-only)."""
    import torch
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_head": None if head is None else head.strip(),
            "git_dirty": None if status is None else [line for line in status.splitlines()
                                                      if line.strip()],
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__}


def resolved_plan(config_path: Path, stage: str, run_root: Path, *,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a stage would read and write; printed by --dry-run."""
    config = load_config(config_path)
    return {"schema_id": SCHEMA_ID, "stage": stage, "dry_run": True,
            "config_path": str(config_path), "run_root": str(run_root),
            "manifest": contract_manifest(config), "provenance": provenance(),
            **(extra or {})}


def write_json(path: Path, payload: Any) -> None:
    """Atomic write (tmp + replace) so a reader never sees half a file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
                         + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def owned(root: Path) -> Iterator[Path]:
    """Single writer per folder (two writers would interleave results); a lock left by a dead
    process is cleared."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
        owner = lock.read_text().strip()
        pid = int(owner.split()[0]) if owner[:1].isdigit() else -1
        alive = pid > 0
        if alive:
            try:
                os.kill(pid, 0)
            except OSError:
                alive = False
        if alive:
            raise SystemExit(f"another run owns {root} (pid {pid}); wait for it or stop it")
        lock.unlink()
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, f"{os.getpid()} {dt.datetime.now().isoformat()}\n".encode())
    os.close(handle)
    try:
        yield root
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()
