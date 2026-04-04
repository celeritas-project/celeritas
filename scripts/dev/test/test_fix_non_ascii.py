# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Tests for scripts/dev/fix-non-ascii.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

# The script has a hyphenated name that isn't a valid Python identifier,
# so we load it explicitly via importlib.
_SCRIPT = Path(__file__).parent.parent / "fix-non-ascii.py"
_spec = importlib.util.spec_from_file_location("fix_non_ascii", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_REPLACEMENTS = _mod._REPLACEMENTS
_UTF8_BOM = _mod._UTF8_BOM
main = _mod.main
process_file = _mod.process_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write(tmp_path, name, content, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(content, encoding=encoding)
    return p


def write_bytes(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# process_file: individual replacement characters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char,expected", _REPLACEMENTS.items())
def test_process_file_replacements(tmp_path, char, expected):
    """Each entry in _REPLACEMENTS is applied correctly."""
    p = write(tmp_path, "test.cc", f"before {char} after")
    modified, errors = process_file(p)
    assert modified
    assert errors == []
    assert p.read_text() == f"before {expected} after"


def test_process_file_multiple_replacements(tmp_path):
    """Multiple replaceable characters in one file are all fixed."""
    p = write(tmp_path, "test.cc", "a\u2192b\u2014c\u2026d")
    modified, errors = process_file(p)
    assert modified
    assert errors == []
    assert p.read_text() == "a->b--c...d"


# ---------------------------------------------------------------------------
# process_file: BOM handling
# ---------------------------------------------------------------------------


def test_process_file_bom_removed(tmp_path):
    """UTF-8 BOM at the start of a file is stripped."""
    p = write_bytes(tmp_path, "bom.cc", _UTF8_BOM + b"clean content")
    modified, errors = process_file(p)
    assert modified
    assert errors == []
    assert p.read_bytes() == b"clean content"


def test_process_file_bom_with_replacement(tmp_path):
    """BOM is stripped and replacements are applied in the same pass."""
    p = write_bytes(tmp_path, "bom.cc", _UTF8_BOM + "arrow\u2192right".encode())
    modified, errors = process_file(p)
    assert modified
    assert errors == []
    assert p.read_bytes() == b"arrow->right"


# ---------------------------------------------------------------------------
# process_file: clean file
# ---------------------------------------------------------------------------


def test_process_file_clean(tmp_path):
    """A file with only ASCII is not modified and produces no errors."""
    content = "// clean ASCII source\nint x = 42;\n"
    p = write(tmp_path, "clean.cc", content)
    modified, errors = process_file(p)
    assert not modified
    assert errors == []
    assert p.read_text() == content


# ---------------------------------------------------------------------------
# process_file: unreplaceable non-ASCII characters
# ---------------------------------------------------------------------------


def test_process_file_unreplaceable_error(tmp_path):
    """Unreplaceable non-ASCII characters are reported with location."""
    p = write(tmp_path, "bad.cc", "r\u00e9sum\u00e9")
    modified, errors = process_file(p)
    assert not modified
    assert len(errors) == 2
    assert "U+00E9" in errors[0]
    assert "bad.cc:1:2" in errors[0]


def test_process_file_unreplaceable_multi_line(tmp_path):
    """Each unreplaceable character is reported with its own line/column."""
    p = write(tmp_path, "bad.cc", "line1\nna\u00efve\nline3")
    modified, errors = process_file(p)
    assert not modified
    assert len(errors) == 1
    assert ":2:3:" in errors[0]
    assert "U+00EF" in errors[0]


def test_process_file_not_written_when_only_errors(tmp_path):
    """File is not rewritten when only unhandled characters are found."""
    original = "r\u00e9sum\u00e9"
    p = write(tmp_path, "bad.cc", original)
    original_mtime = p.stat().st_mtime
    process_file(p)
    assert p.stat().st_mtime == original_mtime


# ---------------------------------------------------------------------------
# main(): exit codes
# ---------------------------------------------------------------------------


def test_main_clean_returns_0(tmp_path):
    p = write(tmp_path, "f.cc", "int main() {}\n")
    assert main([str(p)]) == 0


def test_main_modified_returns_1(tmp_path):
    p = write(tmp_path, "f.cc", "auto x = a\u2192b;")
    assert main([str(p)]) == 1


def test_main_unreplaceable_returns_1(tmp_path):
    p = write(tmp_path, "f.cc", "r\u00e9sum\u00e9")
    assert main([str(p)]) == 1


def test_main_no_files_returns_0():
    assert main([]) == 0
