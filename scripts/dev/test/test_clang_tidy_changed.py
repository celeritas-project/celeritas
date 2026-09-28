# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Tests for scripts/ci/clang-tidy-changed.py."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parents[2] / "ci"
sys.path.insert(0, str(_SCRIPT_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "clang_tidy_changed", _SCRIPT_DIR / "clang-tidy-changed.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
log_compile_commands = _MODULE.log_compile_commands
run_header_tidy = _MODULE.run_header_tidy
run_source_tidy = _MODULE.run_source_tidy


def test_log_compile_commands_reports_missing_sources(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    source = repo_root / "src" / "example.cc"
    missing = repo_root / "test" / "example.test.cc"
    build_dir.mkdir(parents=True)
    (build_dir / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(build_dir),
                    "file": str(source),
                    "command": "clang++ -c example.cc",
                }
            ]
        )
    )

    result = log_compile_commands(
        ["src/example.cc", "test/example.test.cc"], build_dir, repo_root
    )

    assert result == [str(missing)]
    output = capsys.readouterr()
    assert "Compilation database entry: source=src/example.cc" in output.err
    assert "::error file=test/example.test.cc,line=" in output.err
    assert "Source 'test/example.test.cc' selected for clang-tidy" in output.err
    assert "no compilation database entry" in output.err


def test_run_source_tidy_stops_for_missing_source(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "compile_commands.json").write_text("[]")

    result = run_source_tidy(
        Namespace(clang_tidy="clang-tidy", clang_tidy_diff=Path("clang-tidy-diff.py")),
        "diff",
        ["src/missing.cc"],
        repo_root,
        build_dir,
    )

    assert result == 1
    assert "::error file=src/missing.cc,line=" in capsys.readouterr().err


def test_run_header_tidy_stops_for_missing_source(tmp_path, capsys, monkeypatch):
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "compile_commands.json").write_text("[]")
    monkeypatch.setattr(_MODULE, "command_path", lambda command: command)
    monkeypatch.setattr(_MODULE, "scan_dependencies", lambda *args: None)
    monkeypatch.setattr(_MODULE, "select_sources", lambda **kwargs: ["src/missing.cc"])

    result = run_header_tidy(
        Namespace(
            clang_scan_deps="clang-scan-deps",
            clang_tidy="clang-tidy",
            run_clang_tidy="run-clang-tidy",
            header_source_selection=_MODULE.SourceSelection.ALL,
        ),
        ["src/example.hh"],
        [],
        repo_root,
        build_dir,
    )

    assert result == 1
    assert "::error file=src/missing.cc,line=" in capsys.readouterr().err
