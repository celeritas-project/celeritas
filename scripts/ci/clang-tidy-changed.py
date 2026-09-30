#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Run clang-tidy on source changes and translation units affected by headers."""

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from _regression_utils import (
    LogLevel,
    command_path,
    log,
)

SOURCE_PATH_RE = re.compile(r"^(src|app|test)/.*\.(cc|cpp|cu)$")
HEADER_PATH_RE = re.compile(r"^(src|app|test)/.*\.hh$")
DIAGNOSTIC_RE = re.compile(r"^(.*):(\d+):(\d+): error: (.*)$")
HEADER_FILTER_HINT = (
    "Use -header-filter=.* to display errors from all non-system headers."
)
SOURCE_EXT = (".cc", ".cpp", ".cu")


class SourceSelection(StrEnum):
    ALL = "all"
    ONE = "one"


def resolve_paths(paths: Iterable[Path], root: Path) -> set[Path]:
    """Resolve repository-relative paths against the root."""
    return {root.joinpath(path).resolve() for path in paths if path}


def select_sources(
    *,
    header_source_selection: SourceSelection,
    headers: set[Path],
    changed_sources: set[Path],
    dependency_file: Path,
    root: Path,
    build_dir: Path,
    compilation_database: list[dict],
) -> list[str]:
    """Select affected sources, excluding CUDA files without compile commands."""
    if header_source_selection not in SourceSelection:
        raise ValueError(
            f"unsupported header source selection: {header_source_selection!r}"
        )

    root = root.resolve()
    headers = resolve_paths(headers, root)
    changed_sources = resolve_paths(changed_sources, root)
    database_sources = {
        compilation_source(entry, build_dir) for entry in compilation_database
    }

    def selectable(source: Path) -> bool:
        return source.suffix != ".cu" or source in database_sources

    for source in sorted(changed_sources):
        if not selectable(source):
            log(
                LogLevel.NOTICE,
                f"Skipping changed CUDA source without a compilation database entry: "
                f"{source.relative_to(root).as_posix()}",
            )
    changed_sources = {source for source in changed_sources if selectable(source)}
    data = json.loads(dependency_file.read_text())
    affected_sources: set[Path] = set()
    source_by_header: dict[Path, Path] = {}

    commands = (
        command
        for unit in data.get("translation-units", [])
        for command in unit.get("commands", [])
    )
    for command in commands:
        source = command.get("input-file") or command.get("input_file")
        if source is None or not source.endswith(SOURCE_EXT):
            continue
        directory = Path(command.get("directory", root))
        source_path = directory.joinpath(source).resolve()
        if not selectable(source_path):
            continue
        dependencies = command.get("file-deps", command.get("file_deps", []))
        resolved_dependencies = {
            directory.joinpath(dependency).resolve() for dependency in dependencies
        }

        matching_headers = headers & resolved_dependencies
        if header_source_selection is SourceSelection.ALL and matching_headers:
            affected_sources.add(source_path)
        elif header_source_selection is SourceSelection.ONE:
            for header in matching_headers:
                # If multiple sources depend on the same header,
                # pick the one with the lexicographically smallest path.
                source_by_header[header] = min(
                    source_path, source_by_header.get(header, source_path)
                )

    if header_source_selection is SourceSelection.ONE:
        affected_sources = set(source_by_header.values())

    relative_sources = sorted(
        source.relative_to(root).as_posix()
        for source in changed_sources | affected_sources
    )
    return relative_sources


def changed_paths(diff: str) -> tuple[set[Path], set[Path]]:
    """Extract changed headers and sources from a unified diff."""
    paths = [
        line.removeprefix("+++ b/")
        for line in diff.splitlines()
        if line.startswith("+++ b/")
    ]
    return (
        {Path(path) for path in paths if HEADER_PATH_RE.match(path)},
        {Path(path) for path in paths if SOURCE_PATH_RE.match(path)},
    )


def scanner_path(clang_tidy: str) -> str:
    """Find the version-matched clang dependency scanner."""
    tidy_path = Path(command_path(clang_tidy))
    match = re.fullmatch(r"clang-tidy(-.*)?", tidy_path.name)
    suffix = match.group(1) if match is not None else "-18"
    return str(tidy_path.with_name(f"clang-scan-deps{suffix}"))


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
                log(
                    LogLevel.ERROR,
                    message,
                    file=relative_path,
                    line=line_number,
                    col=column,
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
    """Fetch the base commit and diff its tree against the current HEAD."""
    log(LogLevel.NOTICE, f"Fetching base commit {base_sha} from {remote}")
    subprocess.run(
        ["git", "fetch", "--depth", "1", remote, base_sha],
        cwd=repo_root,
        check=True,
    )
    return subprocess.run(
        ["git", "diff", "--diff-filter=ACM", "-U0", f"{base_sha}..HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def load_compilation_database(build_dir: Path) -> list[dict]:
    """Load the compilation database from the build directory."""
    return json.loads((build_dir / "compile_commands.json").read_text())


def compilation_source(entry: dict, build_dir: Path) -> Path:
    """Resolve a compilation database source relative to its working directory."""
    directory = Path(entry.get("directory", build_dir))
    if not directory.is_absolute():
        directory = build_dir / directory
    source = Path(entry["file"])
    return (directory / source).resolve()


def scannable_commands(compilation_database: list[dict], build_dir: Path) -> list[dict]:
    """Exclude nvcc commands, which clang-scan-deps cannot parse."""
    result = []
    for entry in compilation_database:
        command = entry.get("arguments")
        if command is None:
            command = shlex.split(entry.get("command", ""))
        if any(Path(arg).name == "nvcc" for arg in command):
            log(
                LogLevel.NOTICE,
                f"Skipping nvcc dependency-scan source: {compilation_source(entry, build_dir)}",
            )
            continue
        result.append(entry)
    return result


def scan_dependencies(
    scanner: str,
    build_dir: Path,
    repo_root: Path,
    output: Path,
    compilation_database: list[dict],
) -> None:
    """Scan available compile commands, reporting skipped missing inputs."""
    build_dir = build_dir.resolve()
    repo_root = repo_root.resolve()
    scan_database = []
    for entry in compilation_database:
        source_path = compilation_source(entry, build_dir)
        if source_path.is_file():
            scan_database.append(entry)
            continue
        if source_path.name == "CeleritasRootInterface.cxx":
            log(
                LogLevel.NOTICE,
                f"Ignoring missing ROOT dictionary source: {source_path}",
            )
            continue

        try:
            relative_path = source_path.relative_to(repo_root).as_posix()
        except ValueError:
            relative_path = str(source_path)
        properties = (
            {"file": relative_path} if source_path.is_relative_to(repo_root) else {}
        )
        log(
            LogLevel.WARNING,
            f"Skipping unavailable dependency-scan source {relative_path!r}",
            **properties,
        )

    if not scan_database:
        output.write_text(json.dumps({"translation-units": []}))
        return

    scan_database_path = output.with_name("compile_commands.json")
    scan_database_path.write_text(json.dumps(scan_database))
    with output.open("w") as dependency_output:
        subprocess.run(
            [
                scanner,
                "-compilation-database",
                str(scan_database_path),
                "-format",
                "experimental-full",
            ],
            cwd=repo_root,
            check=True,
            stdout=dependency_output,
        )


def source_selector(paths: list[str]) -> str:
    """Create a run-clang-tidy regex that exactly matches source paths."""
    return "(?:^|/)(?:" + "|".join(re.escape(path) for path in paths) + ")$"


def validate_selected_sources(
    sources: list[str],
    build_dir: Path,
    repo_root: Path,
    compilation_database: list[dict],
    *,
    print_commands: bool = False,
) -> list[str]:
    """Validate selected sources and log missing compilation commands.

    Existing sources without compile commands are warned about and skipped;
    nonexistent sources are returned as errors.
    """
    selected_paths = resolve_paths(sources, repo_root)
    matched_sources: set[Path] = set()

    for entry in compilation_database:
        source_path = compilation_source(entry, build_dir)
        if source_path not in selected_paths:
            continue

        if print_commands:
            directory = Path(entry.get("directory", build_dir))
            source = source_path.relative_to(repo_root).as_posix()
            command = entry.get("command", entry.get("arguments"))
            log(
                LogLevel.NOTICE,
                f"Compilation database entry: source={source}; "
                f"directory={directory}; command={command!r}",
            )
        matched_sources.add(source_path)

    unavailable_sources = sorted(
        (selected_paths - matched_sources)
        | {source for source in matched_sources if not source.is_file()}
    )
    missing_sources: list[str] = []
    for source_path in unavailable_sources:
        try:
            relative_path = source_path.relative_to(repo_root).as_posix()
        except ValueError:
            relative_path = str(source_path)
        if not source_path.is_file():
            reason = "does not exist"
            level = LogLevel.ERROR
            missing_sources.append(source_path.as_posix())
        else:
            reason = "has no compilation database entry"
            level = LogLevel.WARNING
        log(
            level,
            f"Source {relative_path!r} selected for clang-tidy {reason}",
            file=relative_path,
        )
    return missing_sources


def sources_with_compilation_commands(
    sources: list[str],
    build_dir: Path,
    repo_root: Path,
    compilation_database: list[dict],
) -> list[str]:
    """Return selected existing sources with compilation database entries."""
    selected_paths = resolve_paths((Path(source) for source in sources), repo_root)
    matched_sources = {
        compilation_source(entry, build_dir) for entry in compilation_database
    }
    return sorted(
        source.relative_to(repo_root).as_posix()
        for source in selected_paths & matched_sources
        if source.is_file()
    )


def run_header_tidy(
    args: argparse.Namespace,
    headers: set[Path],
    sources: set[Path],
    repo_root: Path,
    build_dir: Path,
) -> int:
    """Run clang-tidy on compilation units affected by changed headers."""
    scanner = command_path(args.clang_scan_deps or scanner_path(args.clang_tidy))
    runner = command_path(args.run_clang_tidy)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dependency_file = temp_dir / "dependencies.json"
        compilation_database = scannable_commands(
            load_compilation_database(build_dir), build_dir
        )
        log(LogLevel.NOTICE, "Header changes detected: finding affected source files")
        scan_dependencies(
            scanner, build_dir, repo_root, dependency_file, compilation_database
        )
        selected_sources = select_sources(
            header_source_selection=args.header_source_selection,
            headers=headers,
            changed_sources=sources,
            dependency_file=dependency_file,
            root=repo_root,
            build_dir=build_dir,
            compilation_database=compilation_database,
        )
        if not selected_sources:
            log(LogLevel.NOTICE, "No source files selected for the changed headers")
            return 0
        log(LogLevel.NOTICE, f"Selected {len(selected_sources)} source files:")
        for source in selected_sources:
            log(LogLevel.NOTICE, f"  {source}")
        missing_sources = validate_selected_sources(
            selected_sources,
            build_dir,
            repo_root,
            compilation_database,
            print_commands=args.print_compile_commands,
        )
        if missing_sources:
            raise RuntimeError("missing sources")
        selected_sources = sources_with_compilation_commands(
            selected_sources, build_dir, repo_root, compilation_database
        )
        if not selected_sources:
            log(LogLevel.NOTICE, "No source files with compilation database entries")
            return 0
        log(
            LogLevel.NOTICE,
            f"Running clang-tidy on {len(selected_sources)} affected source files",
        )
        return run_tidy(
            [
                runner,
                "-clang-tidy-binary",
                args.clang_tidy,
                "-p",
                str(build_dir),
                source_selector(selected_sources),
            ],
            repo_root,
        )


def run_source_tidy(
    args: argparse.Namespace,
    diff: str,
    sources: set[Path],
    repo_root: Path,
    build_dir: Path,
) -> int:
    """Run clang-tidy-diff.py for changed .cc files, not CUDA sources."""
    # clang-tidy-diff.py is restricted to .cc files by the regex below.
    for source in sorted(sources):
        if source.suffix == ".cu":
            log(
                LogLevel.NOTICE,
                f"Skipping changed CUDA source in .cc-only clang-tidy-diff run: "
                f"{source.as_posix()}",
            )
    tidy_sources = [source.as_posix() for source in sources if source.suffix == ".cc"]
    if not tidy_sources:
        log(LogLevel.NOTICE, "No .cc source files selected for clang-tidy-diff")
        return 0
    compilation_database = load_compilation_database(build_dir)
    if validate_selected_sources(
        tidy_sources,
        build_dir,
        repo_root,
        compilation_database,
        print_commands=args.print_compile_commands,
    ):
        raise RuntimeError("missing sources")

    tidy_sources = sources_with_compilation_commands(
        tidy_sources, build_dir, repo_root, compilation_database
    )
    if not tidy_sources:
        log(LogLevel.NOTICE, "No .cc source files with compilation database entries")
        return 0

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
            source_selector(tidy_sources),
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
    return run_source_tidy(args, diff, sources, repo_root, build_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and execute the changed-file clang-tidy check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("remote")
    parser.add_argument("base_sha")
    parser.add_argument("--clang-tidy", required=True)
    parser.add_argument("--clang-tidy-diff", type=Path, required=True)
    parser.add_argument("--clang-scan-deps")
    parser.add_argument("--run-clang-tidy", default="run-clang-tidy")
    parser.add_argument("--print-compile-commands", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, default=Path.cwd() / "build")
    parser.add_argument(
        "--header-source-selection",
        type=SourceSelection,
        choices=tuple(SourceSelection),
        default=SourceSelection.ALL,
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    raise SystemExit(main())
