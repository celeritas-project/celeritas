# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Tests for scripts/ci/clang-tidy-changed.py."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).parents[2] / "ci"
sys.path.insert(0, str(_SCRIPT_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "clang_tidy_changed", _SCRIPT_DIR / "clang-tidy-changed.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
log_compile_commands = _MODULE.log_compile_commands
scan_dependencies = _MODULE.scan_dependencies
generated_sources = _MODULE.generated_sources
run_header_tidy = _MODULE.run_header_tidy
run_source_tidy = _MODULE.run_source_tidy


@pytest.fixture
def build_tree(tmp_path):
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True)
    return repo_root, build_dir


@pytest.fixture
def write_codemodel():
    def write(build_dir, targets):
        reply_dir = build_dir / ".cmake/api/v1/reply"
        reply_dir.mkdir(parents=True)
        target_refs = []
        for index, sources in enumerate(targets):
            target_name = f"target-{index}.json"
            (reply_dir / target_name).write_text(json.dumps({"sources": sources}))
            target_refs.append({"jsonFile": target_name})

        (reply_dir / "index-abc.json").write_text(
            json.dumps({"reply": {"codemodel-v2": {"jsonFile": "codemodel.json"}}})
        )
        (reply_dir / "codemodel.json").write_text(
            json.dumps({"configurations": [{"targets": target_refs}]})
        )

    return write


def test_log_compile_commands_reports_missing_sources(build_tree, capsys):
    repo_root, build_dir = build_tree
    source = repo_root / "src" / "example.cc"
    missing = repo_root / "test" / "example.test.cc"
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

    compilation_database = json.loads((build_dir / "compile_commands.json").read_text())
    result = log_compile_commands(
        ["src/example.cc", "test/example.test.cc"],
        build_dir,
        repo_root,
        compilation_database,
    )

    assert result == [str(missing)]
    output = capsys.readouterr()
    assert "Compilation database entry: source=src/example.cc" in output.err
    assert "::error file=test/example.test.cc,line=" in output.err
    assert "Source 'test/example.test.cc' selected for clang-tidy" in output.err
    assert "no compilation database entry" in output.err


@pytest.mark.parametrize("header_mode", [False, True], ids=["changed-source", "header"])
def test_run_tidy_stops_for_missing_source(
    build_tree, capsys, monkeypatch, header_mode
):
    repo_root, build_dir = build_tree
    (build_dir / "compile_commands.json").write_text("[]")
    database_loads = []
    monkeypatch.setattr(
        _MODULE,
        "load_compilation_database",
        lambda _: database_loads.append(True) or [],
    )

    if header_mode:
        monkeypatch.setattr(_MODULE, "command_path", lambda command: command)
        monkeypatch.setattr(_MODULE, "scan_dependencies", lambda *args: True)
        monkeypatch.setattr(
            _MODULE, "select_sources", lambda **kwargs: ["src/missing.cc"]
        )
        result = run_header_tidy(
            Namespace(
                clang_scan_deps="clang-scan-deps",
                clang_tidy="clang-tidy",
                run_clang_tidy="run-clang-tidy",
                header_source_selection=_MODULE.SourceSelection.ALL,
            ),
            {repo_root / "src/example.hh"},
            set(),
            repo_root,
            build_dir,
        )
    else:
        result = run_source_tidy(
            Namespace(
                clang_tidy="clang-tidy",
                clang_tidy_diff=Path("clang-tidy-diff.py"),
            ),
            "diff",
            {Path("src/missing.cc")},
            repo_root,
            build_dir,
        )

    assert result == 1
    assert len(database_loads) == 1
    assert "::error file=src/missing.cc,line=" in capsys.readouterr().err


@pytest.mark.parametrize("is_generated", [True, False])
def test_generated_sources_reads_codemodel(build_tree, write_codemodel, is_generated):
    repo_root, build_dir = build_tree
    source = build_dir / "src" / "Generated.cxx"
    write_codemodel(
        build_dir,
        [[{"path": str(source), "isGenerated": is_generated}]],
    )

    expected = {source.resolve()} if is_generated else set()
    assert generated_sources(build_dir, repo_root) == expected


def test_scan_dependencies_ignores_missing_generated_source(
    build_tree, write_codemodel, monkeypatch
):
    repo_root, build_dir = build_tree
    generated = build_dir / "src" / "Generated.cxx"
    database = build_dir / "compile_commands.json"
    database.write_text(
        json.dumps([{"directory": str(build_dir), "file": str(generated)}])
    )
    write_codemodel(build_dir, [[{"path": str(generated), "isGenerated": True}]])
    scan_command = None

    def fake_run(command, **kwargs):
        nonlocal scan_command
        scan_command = command
        kwargs["stdout"].write('{"translation-units": []}')

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    output = repo_root / "dependencies.json"

    compilation_database = json.loads(database.read_text())
    assert scan_dependencies(
        "clang-scan-deps", build_dir, repo_root, output, compilation_database
    )
    assert scan_command is not None
    scan_database = Path(scan_command[scan_command.index("-compilation-database") + 1])
    assert json.loads(scan_database.read_text()) == []


def test_scan_dependencies_errors_on_missing_unmarked_source(
    build_tree, write_codemodel, capsys
):
    repo_root, build_dir = build_tree
    source = repo_root / "src" / "Unexpected.cc"
    (build_dir / "compile_commands.json").write_text(
        json.dumps([{"directory": str(build_dir), "file": str(source)}])
    )
    write_codemodel(build_dir, [])

    compilation_database = json.loads((build_dir / "compile_commands.json").read_text())
    assert not scan_dependencies(
        "clang-scan-deps",
        build_dir,
        repo_root,
        repo_root / "dependencies.json",
        compilation_database,
    )
    assert "src/Unexpected.cc" in capsys.readouterr().err
