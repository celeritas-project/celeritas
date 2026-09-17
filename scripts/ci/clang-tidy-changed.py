#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Run clang-tidy on source changes and translation units affected by headers."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from _regression_utils import LogLevel, log

SOURCE_PATH_RE = re.compile(r"^(src|app|test)/.*\.(cc|cpp|cu)$")
HEADER_PATH_RE = re.compile(r"^(src|app|test)/.*\.hh$")
DIAGNOSTIC_RE = re.compile(r"^(.*):(\d+):(\d+): error: (.*)$")
HEADER_FILTER_HINT = (
    "Use -header-filter=.* to display errors from all non-system headers."
)


class HeaderSources(StrEnum):
    ALL = "all"
    ONE = "one"


def command_path(command: str) -> str:
    """Resolve an executable name or fail with a helpful error."""
    if path := shutil.which(command):
        return path
    raise RuntimeError(f"executable not found: {command}")


def changed_paths(diff: str) -> tuple[list[str], list[str]]:
    """Extract changed headers and sources from a unified diff."""
    paths = [
        line.removeprefix("+++ b/")
        for line in diff.splitlines()
        if line.startswith("+++ b/")
    ]
    return (
        [path for path in paths if HEADER_PATH_RE.match(path)],
        [path for path in paths if SOURCE_PATH_RE.match(path)],
    )


def scanner_path(clang_tidy: str) -> str:
    """Find the version-matched clang dependency scanner."""
    tidy_path = Path(command_path(clang_tidy))
    match = re.fullmatch(r"clang-tidy(-.*)?", tidy_path.name)
    suffix = match.group(1) if match is not None else "-18"
    return os.environ.get(
        "CLANG_SCAN_DEPS", str(tidy_path.with_name(f"clang-scan-deps{suffix}"))
    )


def escape_property(value: str) -> str:
    """Escape GitHub Actions annotation property values."""
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


def format_tidy_output(lines: Iterable[str], repo_root: Path) -> None:
    """Stream tidy output, emitting one GitHub annotation per error."""
    generated: set[str] = set()
    seen: set[tuple[str, str, str, str]] = set()
    suppress_context = 0
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", repo_root))

    for line in lines:
        line = line.rstrip("\n")
        if re.fullmatch(r"\d+ warnings generated\.", line):
            generated.add(line.split()[0])
            continue
        if match := re.fullmatch(r"Suppressed \d+ warnings \((\d+) in .*", line):
            count = match.group(1)
            if count in generated:
                generated.remove(count)
                line = re.sub(r"^Suppressed ", f"{count} warnings generated; ", line)
                line = line.replace(" warnings (", " suppressed (", 1)
            print(line)
            continue
        if line == HEADER_FILTER_HINT:
            continue
        if match := DIAGNOSTIC_RE.match(line):
            path, line_number, column, message = match.groups()
            try:
                relative_path = str(
                    Path(path).resolve().relative_to(workspace.resolve())
                )
            except ValueError:
                relative_path = path
            key = (relative_path, line_number, column, message)
            if key not in seen:
                seen.add(key)
                print(
                    "::error file="
                    f"{escape_property(relative_path)},line={line_number},col={column}::"
                    f"{escape_message(message)}"
                )
            else:
                suppress_context = 2
            continue
        if suppress_context:
            suppress_context -= 1
            continue
        print(line)


def run_tidy(command: list[str], repo_root: Path) -> int:
    """Run clang-tidy and format its combined output without losing its status."""
    log(LogLevel.DEBUG, f"Running {' '.join(command)!r}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    format_tidy_output(process.stdout, repo_root)
    return process.wait()


def run(args: argparse.Namespace) -> int:
    """Run clang-tidy for the requested base commit."""
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    clang_tidy = os.environ.get("CLANG_TIDY")
    clang_tidy_diff = os.environ.get("CLANG_TIDY_DIFF")
    if not clang_tidy or not clang_tidy_diff:
        raise RuntimeError("CLANG_TIDY and CLANG_TIDY_DIFF must be defined")
    if not Path(clang_tidy_diff).is_file():
        raise RuntimeError(f"clang-tidy-diff.py not found: {clang_tidy_diff}")
    command_path(clang_tidy)

    log(LogLevel.NOTICE, f"Fetching base commit {args.base_sha} from {args.remote}")
    subprocess.run(
        ["git", "fetch", "--depth", "1", args.remote, args.base_sha],
        cwd=repo_root,
        check=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--diff-filter=ACM", "-U0", f"{args.base_sha}...HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    headers, sources = changed_paths(diff)
    if headers:
        scanner = command_path(scanner_path(clang_tidy))
        runner = command_path(os.environ.get("RUN_CLANG_TIDY", "run-clang-tidy"))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            header_file = temp_dir / "headers.txt"
            source_file = temp_dir / "sources.txt"
            dependency_file = temp_dir / "dependencies.json"
            regex_file = temp_dir / "sources.regex"
            header_file.write_text("\n".join(headers) + "\n")
            source_file.write_text("\n".join(sources) + "\n")
            log(
                LogLevel.NOTICE,
                "Header changes detected: finding affected source files",
            )
            with dependency_file.open("w") as output:
                subprocess.run(
                    [
                        scanner,
                        "-compilation-database",
                        str(build_dir / "compile_commands.json"),
                        "-format",
                        "experimental-full",
                    ],
                    cwd=repo_root,
                    check=True,
                    stdout=output,
                )
            selected_count = subprocess.run(
                [
                    sys.executable,
                    str(args.source_selector),
                    args.header_sources,
                    str(header_file),
                    str(source_file),
                    str(dependency_file),
                    str(regex_file),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if selected_count == "0":
                log(LogLevel.NOTICE, "No source files selected for the changed headers")
                return 0
            log(
                LogLevel.NOTICE,
                f"Running clang-tidy on {selected_count} affected source files",
            )
            return run_tidy(
                [
                    runner,
                    "-clang-tidy-binary",
                    clang_tidy,
                    "-p",
                    str(build_dir),
                    regex_file.read_text(),
                ],
                repo_root,
            )

    env = dict(os.environ, PYTHONWARNINGS="ignore::SyntaxWarning")
    return subprocess.run(
        [
            sys.executable,
            clang_tidy_diff,
            "-clang-tidy-binary",
            clang_tidy,
            "-p",
            "1",
            "-path",
            str(build_dir),
            "-regex",
            r"^(src|app|test)/.*\.cc$",
        ],
        cwd=repo_root,
        input=diff,
        text=True,
        env=env,
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and execute the changed-file clang-tidy check."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("remote")
    parser.add_argument("base_sha")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "build")
    parser.add_argument(
        "--header-sources",
        type=HeaderSources,
        choices=tuple(HeaderSources),
        default=os.environ.get("CLANG_TIDY_HEADER_SOURCES", HeaderSources.ALL),
    )
    parser.add_argument(
        "--source-selector",
        type=Path,
        default=script_dir / "clang-tidy-affected-sources.py",
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
