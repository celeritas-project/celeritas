# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import inspect
import subprocess
from enum import StrEnum
from pathlib import Path


class Status(StrEnum):
    FAILURE = "failure"
    CHANGED = "changed"
    SUCCESS = "success"
    CANCELLED = "cancelled"


class LogLevel(StrEnum):
    DEBUG = "debug"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"


def log(level: str | LogLevel, what: str) -> None:
    """Emit a GitHub Actions log annotation with caller filename and line number."""
    if not isinstance(level, LogLevel):
        level = LogLevel(level.lower())

    if level not in LogLevel:
        raise ValueError(f"unsupported log level: {level!r}")

    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    filename = "<unknown>"
    lineno = 0
    if caller is not None:
        filename = caller.f_code.co_filename
        lineno = caller.f_lineno

    msg = str(what).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} file={filename},line={lineno}::{msg}")


def run_git(repo_root: Path, *args: str, check: bool = True) -> str:
    log(LogLevel.DEBUG, f"Calling git {args!r}")
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout
