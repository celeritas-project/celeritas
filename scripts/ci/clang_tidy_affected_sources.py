# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Find source files affected by changed headers in clang-scan-deps output."""

import json
import re
from enum import StrEnum
from pathlib import Path

SOURCE_EXT = (".cc", ".cpp", ".cu")


class SourceSelection(StrEnum):
    ALL = "all"
    ONE = "one"


def resolve_paths(path_file: Path, root: Path) -> set[Path]:
    """Read repository-relative paths and resolve them against the root."""
    return {
        root.joinpath(line.strip()).resolve()
        for line in path_file.read_text().splitlines()
        if line.strip()
    }


def run(
    *,
    header_source_selection: SourceSelection,
    header_file: Path,
    source_file: Path,
    dependency_file: Path,
    regex_file: Path,
    root: Path,
) -> int:
    """Select source files and write a regex accepted by run-clang-tidy."""
    if header_source_selection not in SourceSelection:
        raise ValueError(
            f"unsupported header source selection: {header_source_selection!r}"
        )

    root = root.resolve()
    headers = resolve_paths(header_file, root)
    changed_sources = resolve_paths(source_file, root)
    data = json.loads(dependency_file.read_text())
    affected_sources: set[Path] = set()
    source_by_header: dict[Path, Path] = {}

    for unit in data.get("translation-units", []):
        for command in unit.get("commands", []):
            source = command.get("input-file") or command.get("input_file")
            if source is None:
                continue
            directory = Path(command.get("directory", root))
            source_path = directory.joinpath(source).resolve()
            dependencies = command.get("file-deps", command.get("file_deps", []))
            resolved_dependencies = {
                directory.joinpath(dependency).resolve() for dependency in dependencies
            }
            if source_path.suffix not in SOURCE_EXT:
                continue

            matching_headers = headers & resolved_dependencies
            if header_source_selection is SourceSelection.ALL and matching_headers:
                affected_sources.add(source_path)
            elif header_source_selection is SourceSelection.ONE:
                for header in matching_headers:
                    source_by_header[header] = min(
                        source_path, source_by_header.get(header, source_path)
                    )

    if header_source_selection is SourceSelection.ONE:
        affected_sources = set(source_by_header.values())

    relative_sources = sorted(
        source.relative_to(root).as_posix()
        for source in changed_sources | affected_sources
    )
    if relative_sources:
        regex_file.write_text(
            "(?:^|/)(?:" + "|".join(re.escape(path) for path in relative_sources) + ")$"
        )
    else:
        regex_file.write_text("")
    return len(relative_sources)
