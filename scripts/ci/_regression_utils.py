# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import inspect
import shutil
import subprocess
import sys
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


def command_path(command: str) -> str:
    """Resolve an executable name or fail with a helpful error."""
    if path := shutil.which(command):
        return path
    raise RuntimeError(f"executable not found: {command}")


def escape_property(value: str) -> str:
    """Escape a GitHub Actions annotation property value."""
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def escape_message(value: str) -> str:
    """Escape a GitHub Actions annotation message."""
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def log(level: str | LogLevel, what: str) -> None:
    """Emit a plain notice or GitHub Actions annotation at the caller location."""
    if not isinstance(level, LogLevel):
        level = LogLevel(level.lower())

    if level not in LogLevel:
        raise ValueError(f"unsupported log level: {level!r}")

    if level is LogLevel.NOTICE:
        print(what, file=sys.stderr)
        return

    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    filename = "<unknown>"
    lineno = 0
    if caller is not None:
        filename = caller.f_code.co_filename
        lineno = caller.f_lineno

    msg = str(what).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} file={filename},line={lineno}::{msg}", file=sys.stderr)


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
