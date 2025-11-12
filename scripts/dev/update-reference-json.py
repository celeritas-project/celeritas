#!/usr/bin/env python
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Update reference JSON values in googletest C++ test files based on actual output.

This script reads googletest output from stdin, parses failure messages to
extract actual JSON values, and safely replaces the corresponding expected
values in the source test files.
"""

import itertools
import sys
import re
import os


def parse_gtest_output(output_text):
    """Parse googletest output to extract file paths, line numbers, and actual
    JSON values."""
    failures = []

    # Pattern to match file path and line number
    file_pattern = r"(.*\.cc):(\d+): Failure"

    failures = []
    lines_iter = iter(output_text.split("\n"))

    for line in lines_iter:
        # Check for file path and line number in failure message
        file_match = re.search(file_pattern, line)
        if not file_match:
            continue

        # Found a new failure
        file_path = file_match.group(1)
        line_number = int(file_match.group(2))

        # Skip lines until we find the actual JSON section
        for line in lines_iter:
            if re.match(r"/\*+\s*ACTUAL\s*\*+/", line):
                break

        # Collect actual JSON until the end marker
        actual_section = "".join(
            itertools.takewhile(lambda l: not re.match(r"/\*+/", l), lines_iter)
        )

        # Add the collected failure
        failures.append(
            {
                "file_path": file_path,
                "line_number": line_number,
                "actual_json": actual_section,
            }
        )

    return failures


def group_failures_by_file(failures):
    """Group failures by file path and sort by line number."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for failure in failures:
        grouped[failure["file_path"]].append(failure)

    # Sort failures within each file by line number (descending so we update
    # from bottom up)
    for file_path in grouped:
        grouped[file_path].sort(key=lambda x: x["line_number"])

    return dict(grouped)


def update_test_file_batch(file_path, failures):
    """Update the test file with multiple JSON values in a single read/write operation."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_count = 0

    # Process failures in reverse line order to avoid line number shifts
    for failure in reversed(failures):
        assert failure["file_path"] == file_path
        line_number = failure["line_number"] - 1  # Convert to 0-based index
        line_slc = slice(line_number - 2, line_number + 2)
        actual_json = failure["actual_json"]

        if line_number > len(lines):
            print(
                f"Warning: Line number {line_number} exceeds file length in {file_path}"
            )
            continue

        # Find the line with EXPECT_JSON_EQ (may be at beginning or end of
        # macro
        orig = "".join(lines[line_slc])

        # Match the EXPECT_JSON_EQ pattern with capture groups
        match = re.search(
            r'\s*EXPECT_JSON_EQ\(\s*(R"json\(.*?\)json"|"[^"]*"),\s*', orig
        )
        if not match:
            print(
                f"  Warning: Could not parse EXPECT_JSON_EQ format at line {line_number + 1}"
            )
            continue

        # Replace only the expected JSON part while preserving surrounding code
        new = orig[: match.start(1)] + actual_json.strip() + orig[match.end(1) :]

        if new != orig:
            lines[line_slc] = new
            print(f"  Updated line {line_number}")
            updated_count += 1
        else:
            print(f"  Warning: No changes made at line {line_number}")

    # Write the file only if we made changes
    if updated_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return updated_count


def main():
    """Main function to process stdin and update test files."""
    if sys.stdin.isatty():
        print("Usage: python update-reference-json.py < gtest_output.txt")
        print("This script reads googletest output from stdin and updates test files.")
        return 1

    # Read all input from stdin
    output_text = sys.stdin.read()

    # Parse the googletest output
    failures = parse_gtest_output(output_text)

    if not failures:
        print("No JSON failures found in the input")
        return 0

    print(f"Found {len(failures)} JSON failures to update")

    # Group failures by file to minimize file I/O
    grouped_failures = group_failures_by_file(failures)

    total_updated_count = 0
    for file_path, file_failures in grouped_failures.items():
        print(f"\nProcessing {file_path} ({len(file_failures)} failures)...")
        try:
            updated_count = update_test_file_batch(file_path, file_failures)
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            return 0
        total_updated_count += updated_count

    print(
        f"\nSuccessfully updated {total_updated_count} out of {len(failures)} failures across {len(grouped_failures)} files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
