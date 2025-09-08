#!/usr/bin/env python
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Update reference JSON values in googletest C++ test files based on actual output.

This script reads googletest output from stdin, parses failure messages to
extract actual JSON values, and safely replaces the corresponding expected
values in the source test files.

(NOTE: this script was written using Claude 4.0 via Copilot; its goal is
utility over style.)
"""

import sys
import re
import os


def parse_gtest_output(output_text):
    """Parse googletest output to extract file paths, line numbers, and actual JSON values."""
    failures = []

    # Pattern to match file path and line number
    file_pattern = r'(.*\.cc):(\d+): Failure'

    lines = output_text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for failure location
        file_match = re.search(file_pattern, line)
        if file_match:
            file_path = file_match.group(1)
            line_number = int(file_match.group(2))

            # Look ahead for the actual JSON value
            j = i + 1
            actual_json = None
            is_empty_expected = False

            # Check if this is an "expected is an empty string" case
            while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                if 'expected is an empty string' in lines[j]:
                    is_empty_expected = True
                elif re.match(r'/\**\s*ACTUAL\s*\*+/', lines[j]):
                    # Found the start of actual section, collect until /******/
                    actual_section = []
                    j += 1
                    while j < len(lines) and '/******/' not in lines[j]:
                        actual_section.append(lines[j])
                        j += 1

                    actual_text = '\n'.join(actual_section)

                    # Extract JSON from R"json(...)json" format
                    json_match = re.search(r'R"json\((.*?)\)json"', actual_text, re.DOTALL)
                    if json_match:
                        actual_json = json_match.group(1)
                        break
                j += 1

            if actual_json is not None:
                failures.append({
                    'file_path': file_path,
                    'line_number': line_number,
                    'actual_json': actual_json,
                    'is_empty_expected': is_empty_expected
                })

        i += 1

    return failures


def update_test_file(file_path, line_number, actual_json, is_empty_expected):
    """Update the test file with the actual JSON value at the specified line."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number > len(lines):
            print(f"Warning: Line number {line_number} exceeds file length in {file_path}")
            return False

        # Find the line with EXPECT_JSON_EQ (may be on the exact line or nearby)
        target_line_idx = None
        search_start = max(0, line_number - 3)
        search_end = min(len(lines), line_number + 3)

        for i in range(search_start, search_end):
            if 'EXPECT_JSON_EQ' in lines[i]:
                target_line_idx = i
                break

        if target_line_idx is None:
            print(f"Warning: Could not find EXPECT_JSON_EQ near line {line_number} in {file_path}")
            return False

        # Update the line with the new JSON value
        line = lines[target_line_idx]

        if is_empty_expected:
            # Replace empty string with actual JSON
            new_line = re.sub(
                r'EXPECT_JSON_EQ\s*\(\s*""\s*,',
                f'EXPECT_JSON_EQ(R"json({actual_json})json",',
                line
            )
        else:
            # Replace existing R"json(...)json" with new value
            new_line = re.sub(
                r'R"json\(.*?\)json"',
                f'R"json({actual_json})json"',
                line
            )

        if new_line != line:
            lines[target_line_idx] = new_line

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"Updated {file_path}:{line_number}")
            return True
        else:
            print(f"Warning: No changes made to {file_path}:{line_number}")
            return False

    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


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

    updated_count = 0
    current_file = None
    for failure in failures:
        file_path = failure['file_path']
        if file_path != current_file:
            print(f"Processing {file_path}...")
            current_file = file_path
        if update_test_file(
            file_path,
            failure['line_number'],
            failure['actual_json'],
            failure['is_empty_expected']
        ):
            updated_count += 1

    print(f"\nSuccessfully updated {updated_count} out of {len(failures)} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())
