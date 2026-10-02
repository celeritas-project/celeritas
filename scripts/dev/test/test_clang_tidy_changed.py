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
validate_selected_sources = _MODULE.validate_selected_sources
select_sources = _MODULE.select_sources
scan_dependencies = _MODULE.scan_dependencies
run_header_tidy = _MODULE.run_header_tidy
run_source_tidy = _MODULE.run_source_tidy


@pytest.fixture
def build_tree(tmp_path):
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True)
    return repo_root, build_dir


def test_validate_selected_sources_reports_missing_entries(build_tree, capsys):
    repo_root, build_dir = build_tree
    source = repo_root / "src" / "example.cc"
    missing = repo_root / "test" / "example.test.cc"
    source.parent.mkdir()
    source.touch()
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
    result = validate_selected_sources(
        ["src/example.cc", "test/example.test.cc"],
        build_dir,
        repo_root,
        compilation_database,
    )

    assert result == [str(missing)]
    output = capsys.readouterr()
    assert "Compilation database entry:" not in output.err
    assert "::error file=test/example.test.cc,line=" in output.err
    assert "Source 'test/example.test.cc' selected for clang-tidy" in output.err
    assert "no compilation database entry" in output.err


def test_validate_selected_sources_prints_commands_when_requested(build_tree, capsys):
    repo_root, build_dir = build_tree
    source = repo_root / "src" / "example.cc"
    source.parent.mkdir()
    source.touch()
    compilation_database = [
        {
            "directory": str(build_dir),
            "file": str(source),
            "command": "clang++ -c example.cc",
        }
    ]

    assert not validate_selected_sources(
        ["src/example.cc"],
        build_dir,
        repo_root,
        compilation_database,
        print_commands=True,
    )
    assert (
        "Compilation database entry: source=src/example.cc;" in capsys.readouterr().err
    )


def test_validate_selected_sources_reports_missing_files(build_tree, capsys):
    repo_root, build_dir = build_tree
    missing = repo_root / "src" / "missing.cc"
    compilation_database = [
        {"directory": str(build_dir), "file": str(missing), "command": "clang++"}
    ]

    assert validate_selected_sources(
        ["src/missing.cc"], build_dir, repo_root, compilation_database
    ) == [str(missing)]
    output = capsys.readouterr().err
    assert "::error file=src/missing.cc,line=" in output
    assert "selected for clang-tidy does not exist" in output


@pytest.mark.parametrize("mode", list(_MODULE.SourceSelection))
@pytest.mark.parametrize("cuda_in_database", [False, True])
def test_select_sources_cuda_requires_compilation_entry(
    build_tree, tmp_path, capsys, mode, cuda_in_database
):
    repo_root, build_dir = build_tree
    header = repo_root / "src" / "example.hh"
    cuda = repo_root / "src" / "a.cu"
    cpp = repo_root / "src" / "b.cc"
    dependency_file = tmp_path / "dependencies.json"
    dependency_file.write_text(
        json.dumps(
            {
                "translation-units": [
                    {
                        "commands": [
                            {"input-file": str(source), "file-deps": [str(header)]}
                            for source in (cuda, cpp)
                        ]
                    }
                ]
            }
        )
    )
    database = [{"directory": str(build_dir), "file": str(cpp)}]
    if cuda_in_database:
        database.append({"directory": str(build_dir), "file": str(cuda)})

    result = select_sources(
        header_source_selection=mode,
        headers={header},
        changed_sources={cuda},
        dependency_file=dependency_file,
        root=repo_root,
        build_dir=build_dir,
        compilation_database=database,
    )

    expected = (
        ["src/a.cu"]
        if mode is _MODULE.SourceSelection.ONE
        else ["src/a.cu", "src/b.cc"]
    )
    assert result == (expected if cuda_in_database else ["src/b.cc"])
    assert ("Skipping changed CUDA source" in capsys.readouterr().err) == (
        not cuda_in_database
    )


@pytest.mark.parametrize("mode", list(_MODULE.SourceSelection))
def test_select_sources_prefers_available_cc_over_unavailable_cuda(
    build_tree, tmp_path, mode
):
    repo_root, build_dir = build_tree
    header = repo_root / "src" / "example.hh"
    dependency_file = tmp_path / "dependencies.json"
    dependency_file.write_text(
        json.dumps(
            {
                "translation-units": [
                    {
                        "commands": [
                            {"input-file": "src/a.cu", "file-deps": [str(header)]},
                            {"input-file": "src/b.cc", "file-deps": [str(header)]},
                        ]
                    }
                ]
            }
        )
    )

    assert select_sources(
        header_source_selection=mode,
        headers={header},
        changed_sources=set(),
        dependency_file=dependency_file,
        root=repo_root,
        build_dir=build_dir,
        compilation_database=[{"directory": str(build_dir), "file": "src/b.cc"}],
    ) == ["src/b.cc"]


def test_run_source_tidy_skips_cuda_and_checks_cc(build_tree, monkeypatch, capsys):
    repo_root, build_dir = build_tree
    cpp = repo_root / "src" / "example.cc"
    cpp.parent.mkdir()
    cpp.touch()
    (build_dir / "compile_commands.json").write_text(
        json.dumps([{"directory": str(build_dir), "file": str(cpp)}])
    )
    calls = []
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Namespace(returncode=0),
    )

    assert (
        run_source_tidy(
            Namespace(
                clang_tidy="clang-tidy",
                clang_tidy_diff=Path("clang-tidy-diff.py"),
                print_compile_commands=False,
            ),
            "diff",
            {Path("src/example.cc"), Path("src/example.cu")},
            repo_root,
            build_dir,
        )
        == 0
    )
    assert len(calls) == 1
    assert "Skipping changed CUDA source in .cc-only" in capsys.readouterr().err


def test_run_source_tidy_cuda_only_skips_runner(build_tree, monkeypatch, capsys):
    repo_root, build_dir = build_tree

    def unexpected_run(*args, **kwargs):
        pytest.fail("clang-tidy-diff should not run for CUDA-only changes")

    monkeypatch.setattr(_MODULE.subprocess, "run", unexpected_run)
    assert (
        run_source_tidy(
            Namespace(), "diff", {Path("src/example.cu")}, repo_root, build_dir
        )
        == 0
    )
    output = capsys.readouterr().err
    assert "Skipping changed CUDA source in .cc-only" in output
    assert "No .cc source files selected" in output


@pytest.mark.parametrize("header_mode", [False, True], ids=["changed-source", "header"])
@pytest.mark.parametrize("has_entry", [False, True], ids=["no-entry", "missing-file"])
def test_run_tidy_stops_for_missing_source(
    build_tree, capsys, monkeypatch, header_mode, has_entry
):
    repo_root, build_dir = build_tree
    missing = repo_root / "src" / "missing.cc"
    database = (
        [{"directory": str(build_dir), "file": str(missing)}] if has_entry else []
    )
    (build_dir / "compile_commands.json").write_text(json.dumps(database))
    database_loads = []
    monkeypatch.setattr(
        _MODULE,
        "load_compilation_database",
        lambda _: database_loads.append(True) or database,
    )

    if header_mode:
        monkeypatch.setattr(_MODULE, "command_path", lambda command: command)
        monkeypatch.setattr(_MODULE, "scan_dependencies", lambda *args: True)
        monkeypatch.setattr(
            _MODULE, "select_sources", lambda **kwargs: ["src/missing.cc"]
        )
        with pytest.raises(RuntimeError, match="^missing sources$"):
            run_header_tidy(
                Namespace(
                    clang_scan_deps="clang-scan-deps",
                    clang_tidy="clang-tidy",
                    run_clang_tidy="run-clang-tidy",
                    header_source_selection=_MODULE.SourceSelection.ALL,
                    print_compile_commands=False,
                ),
                {repo_root / "src/example.hh"},
                set(),
                repo_root,
                build_dir,
            )
    else:
        with pytest.raises(RuntimeError, match="^missing sources$"):
            run_source_tidy(
                Namespace(
                    clang_tidy="clang-tidy",
                    clang_tidy_diff=Path("clang-tidy-diff.py"),
                    print_compile_commands=False,
                ),
                "diff",
                {Path("src/missing.cc")},
                repo_root,
                build_dir,
            )

    assert len(database_loads) == 1
    assert "::error file=src/missing.cc,line=" in capsys.readouterr().err


def test_run_header_tidy_skips_unrelated_missing_scan_input(
    build_tree, monkeypatch, capsys
):
    repo_root, build_dir = build_tree
    selected = repo_root / "src" / "example.cc"
    selected.parent.mkdir()
    selected.touch()
    unrelated = repo_root / "src" / "unrelated.cc"
    (build_dir / "compile_commands.json").write_text(
        json.dumps(
            [
                {"directory": str(build_dir), "file": str(selected)},
                {"directory": str(build_dir), "file": str(unrelated)},
            ]
        )
    )
    monkeypatch.setattr(_MODULE, "command_path", lambda command: command)
    monkeypatch.setattr(_MODULE, "select_sources", lambda **kwargs: ["src/example.cc"])
    monkeypatch.setattr(_MODULE, "run_tidy", lambda *args: 0)

    def fake_run(command, **kwargs):
        kwargs["stdout"].write('{"translation-units": []}')

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert (
        run_header_tidy(
            Namespace(
                clang_scan_deps="clang-scan-deps",
                run_clang_tidy="run-clang-tidy",
                clang_tidy="clang-tidy",
                header_source_selection=_MODULE.SourceSelection.ALL,
                print_compile_commands=False,
            ),
            {Path("src/example.hh")},
            set(),
            repo_root,
            build_dir,
        )
        == 0
    )
    assert "Skipping unavailable dependency-scan source 'src/unrelated.cc'" in (
        capsys.readouterr().err
    )


def test_main_prints_runtime_error_and_exits(monkeypatch, capsys):
    def fail(_):
        raise RuntimeError("missing sources")

    monkeypatch.setattr(_MODULE, "run", fail)
    with pytest.raises(SystemExit) as exc:
        _MODULE.main(
            [
                "origin",
                "base",
                "--clang-tidy",
                "clang-tidy",
                "--clang-tidy-diff",
                "clang-tidy-diff.py",
            ]
        )

    assert exc.value.code == 1
    assert capsys.readouterr().err == "error: missing sources\n"


@pytest.mark.parametrize("print_commands", [False, True])
def test_main_print_compile_commands_option(monkeypatch, print_commands):
    seen = []
    monkeypatch.setattr(
        _MODULE, "run", lambda args: seen.append(args.print_compile_commands) or 0
    )
    argv = [
        "origin",
        "base",
        "--clang-tidy",
        "clang-tidy",
        "--clang-tidy-diff",
        "clang-tidy-diff.py",
    ]
    if print_commands:
        argv.append("--print-compile-commands")

    assert _MODULE.main(argv) == 0
    assert seen == [print_commands]


def test_scan_dependencies_ignores_missing_root_dictionary(build_tree, monkeypatch):
    repo_root, build_dir = build_tree
    generated = build_dir / "src" / "CeleritasRootInterface.cxx"
    existing = repo_root / "src" / "example.cc"
    existing.parent.mkdir()
    existing.touch()
    database = build_dir / "compile_commands.json"
    database.write_text(
        json.dumps(
            [
                {"directory": str(build_dir), "file": str(generated)},
                {"directory": str(build_dir), "file": str(existing)},
            ]
        )
    )
    scan_command = None

    def fake_run(command, **kwargs):
        nonlocal scan_command
        scan_command = command
        kwargs["stdout"].write('{"translation-units": []}')

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    output = repo_root / "dependencies.json"

    compilation_database = json.loads(database.read_text())
    scan_dependencies(
        "clang-scan-deps", build_dir, repo_root, output, compilation_database
    )
    assert scan_command is not None
    scan_database = Path(scan_command[scan_command.index("-compilation-database") + 1])
    assert json.loads(scan_database.read_text()) == [compilation_database[1]]


def test_scan_dependencies_skips_other_missing_source(build_tree, capsys, monkeypatch):
    repo_root, build_dir = build_tree
    source = repo_root / "src" / "Unexpected.cxx"
    (build_dir / "compile_commands.json").write_text(
        json.dumps([{"directory": str(build_dir), "file": str(source)}])
    )

    compilation_database = json.loads((build_dir / "compile_commands.json").read_text())

    def unexpected_run(*args, **kwargs):
        pytest.fail("scanner should not run with an empty compilation database")

    monkeypatch.setattr(_MODULE.subprocess, "run", unexpected_run)
    scan_dependencies(
        "clang-scan-deps",
        build_dir,
        repo_root,
        repo_root / "dependencies.json",
        compilation_database,
    )
    assert "src/Unexpected.cxx" in capsys.readouterr().err
    assert json.loads((repo_root / "dependencies.json").read_text()) == {
        "translation-units": []
    }
