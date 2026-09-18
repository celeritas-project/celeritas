#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Run clang-tidy on source changes and translation units affected by headers."""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from _regression_utils import LogLevel, log
from clang_tidy_affected_sources import SourceSelection, run as select_sources

SOURCE_PATH_RE = re.compile(r"^(src|app|test)/.*\.(cc|cpp|cu)$")
HEADER_PATH_RE = re.compile(r"^(src|app|test)/.*\.hh$")
DIAGNOSTIC_RE = re.compile(r"^(.*):(\d+):(\d+): error: (.*)$")
HEADER_FILTER_HINT = (
    "Use -header-filter=.* to display errors from all non-system headers."
)


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
    return str(tidy_path.with_name(f"clang-scan-deps{suffix}"))


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
                    Path(path).resolve().relative_to(repo_root.resolve())
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


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate filesystem inputs and return resolved repository paths."""
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    if not repo_root.is_dir():
        raise RuntimeError(f"repository root is not a directory: {repo_root}")
    if not build_dir.is_dir():
        raise RuntimeError(f"build directory is not a directory: {build_dir}")
    if not (build_dir / "compile_commands.json").is_file():
        raise RuntimeError(f"compilation database not found in: {build_dir}")
    if not args.clang_tidy_diff.is_file():
        raise RuntimeError(f"clang-tidy-diff.py not found: {args.clang_tidy_diff}")
    command_path(args.clang_tidy)
    return repo_root, build_dir


def fetch_diff(remote: str, base_sha: str, repo_root: Path) -> str:
    """Fetch the base commit and return its diff with the current HEAD."""
    log(LogLevel.NOTICE, f"Fetching base commit {base_sha} from {remote}")
    subprocess.run(
        ["git", "fetch", "--depth", "1", remote, base_sha],
        cwd=repo_root,
        check=True,
    )
    return subprocess.run(
        ["git", "diff", "--diff-filter=ACM", "-U0", f"{base_sha}...HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def scan_dependencies(
    scanner: str, build_dir: Path, repo_root: Path, output: Path
) -> None:
    """Write LLVM's full compilation dependency data to an output file."""
    with output.open("w") as dependency_output:
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
            stdout=dependency_output,
        )


def run_header_tidy(
    args: argparse.Namespace,
    headers: list[str],
    sources: list[str],
    repo_root: Path,
    build_dir: Path,
) -> int:
    """Run clang-tidy on compilation units affected by changed headers."""
    scanner = command_path(args.clang_scan_deps or scanner_path(args.clang_tidy))
    runner = command_path(args.run_clang_tidy)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        header_file = temp_dir / "headers.txt"
        source_file = temp_dir / "sources.txt"
        dependency_file = temp_dir / "dependencies.json"
        regex_file = temp_dir / "sources.regex"
        header_file.write_text("\n".join(headers) + "\n")
        source_file.write_text("\n".join(sources) + "\n")
        log(LogLevel.NOTICE, "Header changes detected: finding affected source files")
        scan_dependencies(scanner, build_dir, repo_root, dependency_file)
        selected_count = select_sources(
            mode=args.header_sources,
            header_file=header_file,
            source_file=source_file,
            dependency_file=dependency_file,
            regex_file=regex_file,
            root=repo_root,
        )
        if selected_count == 0:
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
                args.clang_tidy,
                "-p",
                str(build_dir),
                regex_file.read_text(),
            ],
            repo_root,
        )


def run_source_tidy(
    args: argparse.Namespace, diff: str, repo_root: Path, build_dir: Path
) -> int:
    """Run clang-tidy-diff.py for changed source files only."""
    return subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::SyntaxWarning",
            str(args.clang_tidy_diff),
            "-clang-tidy-binary",
            args.clang_tidy,
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
    ).returncode


def run(args: argparse.Namespace) -> int:
    """Run clang-tidy against sources or header-affected translation units."""
    repo_root, build_dir = validate_inputs(args)
    diff = fetch_diff(args.remote, args.base_sha, repo_root)
    headers, sources = changed_paths(diff)
    if headers:
        return run_header_tidy(args, headers, sources, repo_root, build_dir)
    return run_source_tidy(args, diff, repo_root, build_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and execute the changed-file clang-tidy check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("remote")
    parser.add_argument("base_sha")
    parser.add_argument("--clang-tidy", required=True)
    parser.add_argument("--clang-tidy-diff", type=Path, required=True)
    parser.add_argument("--clang-scan-deps")
    parser.add_argument("--run-clang-tidy", default="run-clang-tidy")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "build")
    parser.add_argument(
        "--header-sources",
        type=SourceSelection,
        choices=tuple(SourceSelection),
        default=SourceSelection.ALL,
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
