#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Fix non-ASCII characters in source files.

Replace known Unicode symbols with ASCII equivalents and remove the UTF-8
byte order mark (BOM). Report an error for any remaining non-ASCII character.

Exit with status 1 if any file was modified or if unrecognized non-ASCII
characters were found.

When invoked from pre-commit, file selection is handled by the ``files:``
pattern in the hook configuration.

Usage (standalone)::

    python3 scripts/dev/fix-non-ascii.py src/celeritas/Types.hh

Usage (pre-commit)::

    pre-commit run fix-non-ascii
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Replacement table
# ---------------------------------------------------------------------------

# UTF-8 encoding of the byte order mark
_UTF8_BOM = b"\xef\xbb\xbf"

# Mapping of unicode character to ASCII replacement.
_REPLACEMENTS = {
    "\u00d7": "x",  # × MULTIPLICATION SIGN
    "\u2010": "-",  # ‐ HYPHEN
    "\u2011": "-",  # ‑ NON-BREAKING HYPHEN
    "\u2012": "-",  # ‒ FIGURE DASH
    "\u2013": "-",  # – EN DASH
    "\u2014": "--",  # — EM DASH
    "\u2018": "'",  # ' LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # ' RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',  # " LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # " RIGHT DOUBLE QUOTATION MARK
    "\u2026": "...",  # … HORIZONTAL ELLIPSIS
    "\u2190": "<-",  # ← LEFT ARROW
    "\u2192": "->",  # → RIGHT ARROW
    "\u21d0": "<=",  # ⇐ LEFTWARDS DOUBLE ARROW
    "\u21d2": "=>",  # ⇒ RIGHTWARDS DOUBLE ARROW
    "\u21d4": "<=>",  # ⇔ LEFT RIGHT DOUBLE ARROW
    "\u2212": "-",  # − MINUS SIGN
}

# C-level translation table: maps each replaceable codepoint to its ASCII string.
_REPLACEMENTS_TABLE = str.maketrans({ord(k): v for k, v in _REPLACEMENTS.items()})

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _find_non_ascii_errors(path, text):
    """Return a list of error strings for any non-ASCII characters in *text*."""
    errors = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for col, ch in enumerate(line, 1):
            if ord(ch) > 127:
                errors.append(
                    f"{path}:{lineno}:{col}: "
                    f"non-ASCII character U+{ord(ch):04X} ({ch!r})"
                )
    return errors


def process_file(path):
    """Fix non-ASCII characters in *path* in place.

    Returns ``(was_modified, error_messages)``.  *was_modified* is True when
    the file was rewritten.  *error_messages* is a (possibly empty) list of
    strings describing non-ASCII characters that could not be replaced.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return False, [f"{path}: {exc}"]

    bom_stripped = raw.startswith(_UTF8_BOM)
    data = raw[3:] if bom_stripped else raw

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, [f"{path}: cannot decode as UTF-8: {exc}"]

    text = text.translate(_REPLACEMENTS_TABLE)

    new_raw = text.encode("utf-8")
    modified = bom_stripped or (new_raw != data)

    if text.isascii():
        errors = []
    else:
        errors = _find_non_ascii_errors(path, text)

    if modified:
        path.write_bytes(new_raw)

    return modified, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        metavar="FILE",
        help="source files to check and fix",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    result = 0
    for filename in args.filenames:
        path = Path(filename)

        modified, errors = process_file(path)

        if modified:
            print(f"Fixed non-ASCII characters in {filename}")
            result = 1

        for error in errors:
            print(error, file=sys.stderr)
        if errors:
            result = 1

    return result


if __name__ == "__main__":
    sys.exit(main())
