#!/usr/bin/env python
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#!/usr/bin/env python3
"""
A Python script that runs cloc on a given git commit in a source directory,
processes its CSV output into pandas tables, and concatenates the results with
section header rows. The script uses fixed cloc flags --csv and --quiet.

Usage:
    python script.py [--source_dir PATH] [--commit COMMIT] [--cloc_path PATH]

If --source_dir is not provided, it defaults to the top-level git directory of the script.
If --cloc_path is not provided, it defaults to 'cloc' (available in PATH).
"""

import os
import sys
import argparse
import subprocess
import shutil
from io import StringIO
from pathlib import Path

import pandas as pd

LANG_MAP = {
    "hip": "CUDA",
}


class RunCloc:
    """Class to execute cloc with configured language mappings and base options."""

    def __init__(self, commit, cloc_path="cloc", cwd=None):
        """
        Initialize with the git commit to analyze and optional language mappings.

        Args:
            commit: Git commit to analyze
            cloc_path: Path to the cloc executable
            lang_map: Dictionary mapping file extensions to language names
        """
        self.commit = commit

        # Validate cloc path
        if isinstance(cloc_path, str):
            cloc_path = shutil.which(cloc_path)
        self.cloc_path = Path(cloc_path)
        if not self.cloc_path.exists():
            sys.stderr.write("This script requires https://github.com/AlDanial/cloc\n")
            raise RuntimeError(f"Cloc executable not found at: {self.cloc_path}")

        self.cwd = cwd

        # Build base command with language mappings
        self.base_cmd = [str(self.cloc_path), "--git", commit, "--csv", "--quiet"]

        # Add language mappings as force-lang options
        for ext, lang in LANG_MAP.items():
            self.base_cmd.append(f"--force-lang={lang},{ext}")

    def _validate_cloc(self):
        """Validate that the cloc executable exists and is executable"""

        return self.cloc_path.exists() and os.access(self.cloc_path, os.X_OK)

    def __call__(self, extra_options):
        """
        Run cloc with the base command plus any additional options.

        Args:
            extra_options: List of additional cloc command line options

        Returns:
            String containing the CSV output from cloc
        """
        command = self.base_cmd.copy()
        command.extend(extra_options)
        return subprocess.check_output(command, text=True, cwd=self.cwd)


def process_csv_output(csv_output):
    """
    Processes cloc's CSV output into a pandas DataFrame.
    Drops the annotation column, removes the sum row, and reorders/renames columns to:
    Language, Files, Comment, Code.
    """
    # Read the CSV using only the first 5 columns (drop the annotation column)
    # Expected CSV header: files,language,blank,comment,code,...
    df = pd.read_csv(StringIO(csv_output)).fillna(0)
    # Remove the sum row
    if df.iloc[-1, 1] == "SUM":
        df = df.iloc[:-1]
    # Rename and reorder columns
    df = df.rename(columns=str.capitalize)[["Language", "Files", "Comment", "Code", "Blank"]]
    # Coerce all but the first column to integers
    for col in df.columns[1:]:
        df[col] = df[col].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run cloc on a git commit and process its CSV output into concatenated pandas tables."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Source directory (defaults to the top-level git directory where the script resides)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="HEAD",
        help="Git commit to run cloc against (default: HEAD)",
    )
    parser.add_argument(
        "--cloc_path",
        type=str,
        default="cloc",
        help="Path to the cloc executable (default: cloc)",
    )
    args = parser.parse_args()

    # Determine the source directory:
    if args.source_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            args.source_dir = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=script_dir, text=True
            ).strip()
        except subprocess.CalledProcessError:
            sys.stderr.write(
                "Unable to determine git top-level directory. Please specify --source_dir\n"
            )
            sys.exit(1)

    # Create the cloc runner
    run_cloc = RunCloc(args.commit, args.cloc_path, args.source_dir)

    # Define each section with its label and specific cloc options.
    EXCLUDE_DOC = "--exclude-ext=rst,md,tex"
    sections = [
        ("Library code", ["--exclude-dir=app,example,scripts,test", EXCLUDE_DOC]),
        ("App code", ["--fullpath", "--match-d=/app/", EXCLUDE_DOC]),
        ("Example code", ["--fullpath", "--match-d=/example/", EXCLUDE_DOC]),
        ("Test code", ["--fullpath", "--match-d=/test/", EXCLUDE_DOC]),
        ("Documentation", [r"--match-f=\.(rst|md|tex)$"]),
    ]

    final_tables = []
    for section_label, options in sections:
        # Print section info to stderr.
        sys.stderr.write(f"{section_label}...\n")
        # Run cloc with the options for this section.
        csv_output = run_cloc(options)
        # Process the CSV output into a DataFrame.
        df_section = process_csv_output(csv_output)
        # Create a header row DataFrame with the section label.
        header_row = pd.DataFrame(
            [{"Language": section_label, "Files": "", "Comment": "", "Code": ""}]
        )
        # Append the header row and the section's table to the final list.
        final_tables.append(header_row)
        final_tables.append(df_section)

    # Concatenate all tables vertically.
    final_df = pd.concat(final_tables, ignore_index=True)
    # Output the final DataFrame as CSV to stdout.
    print(final_df.to_csv(index=False))


if __name__ == "__main__":
    main()
