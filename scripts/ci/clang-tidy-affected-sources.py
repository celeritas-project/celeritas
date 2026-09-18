# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import argparse
from collections.abc import Sequence
from pathlib import Path

from clang_tidy_affected_sources import SourceSelection, run


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and print the count from the source-selection module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "header-source-selection",
        type=SourceSelection,
        choices=tuple(SourceSelection),
    )
    parser.add_argument("header_file", type=Path)
    parser.add_argument("source_file", type=Path)
    parser.add_argument("dependency_file", type=Path)
    parser.add_argument("regex_file", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(
        run(
            header_source_selection=args.header_source_selection,
            header_file=args.header_file,
            source_file=args.source_file,
            dependency_file=args.dependency_file,
            regex_file=args.regex_file,
            root=args.root.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
