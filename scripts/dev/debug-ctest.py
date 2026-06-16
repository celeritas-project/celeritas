#!/usr/bin/env python
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Update .vscode/launch.json to debug a specific CTest test.

Usage::

    set-launch-from-ctest.py [--workspace <dir>] [--build-dir <dir>] <test-name>

The test name can be a full CTest name (e.g.
``accel/UserActionIntegration:LarSphereOpticalOffload.run:g4:mt``) or a
substring that uniquely matches one test.  The first configuration in
``.vscode/launch.json`` is replaced with settings extracted from CTest.

NOTE: this script currently works only for LLDB (LLVM debugger). It can be
updated in the future to work for GDB as well.

ALSO NOTE: the VScode debugger choice ``lldb-dap`` requires an extension to
work: see
https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LLDB_COMMANDS = """
command script import ${workspaceFolder}/scripts/dev/celerlldb.py --allow-reload
type synthetic add -x "^celeritas::Span<.+>$" --python-class celerlldb.SpanSynthetic
type synthetic add -x "^celeritas::Array<.+>$" --python-class celerlldb.ArraySynthetic
type synthetic add -x "^celeritas::ItemRange<.+>$" --python-class celerlldb.ItemRangeSynthetic
type summary add -x "^celeritas::OpaqueId<.+>$" --python-function celerlldb.opaqueid_summary
break set --name celeritas::throw_debug_error
break set --name celeritas::throw_runtime_error
""".strip().splitlines()


def find_workspace_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return Path(result.stdout.strip())
    # Fall back: scripts/dev/ -> project root
    return Path(__file__).resolve().parent.parent.parent


def find_test(build_dir, test_name):
    result = subprocess.run(
        ["ctest", "--show-only=json-v1", "--test-dir", str(build_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"error: ctest failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    matches = [t for t in data["tests"] if test_name in t["name"]]
    if not matches:
        sys.exit(f"error: no test matching {test_name!r}")
    if len(matches) > 1:
        names = "\n  ".join(t["name"] for t in matches)
        sys.exit(f"error: multiple tests match {test_name!r}:\n  {names}")
    return matches[0]


def rel_path(path_str, workspace):
    """Replace an absolute path under the workspace with ${workspaceFolder}/..."""
    ws = str(workspace)
    if path_str.startswith(ws + "/"):
        return "${workspaceFolder}/" + path_str[len(ws) + 1 :]
    return path_str


def make_config(test, workspace):
    command = test["command"]
    props = {p["name"]: p["value"] for p in test.get("properties", [])}
    cwd = props.get("WORKING_DIRECTORY", "")
    env = {}
    for item in props.get("ENVIRONMENT", []):
        k, sep, v = item.partition("=")
        if sep:
            env[k] = v

    config = {
        "type": "lldb-dap",
        "request": "launch",
        "name": test["name"],
        "program": rel_path(command[0], workspace),
        "args": command[1:],
        "cwd": rel_path(cwd, workspace),
        "sourceMap": {"./": "${workspaceFolder}/"},
        "initCommands": LLDB_COMMANDS,
    }
    if env:
        config["env"] = env
    return config


def strip_jsonc_comments(lines):
    """Strip // line comments from JSONC, respecting string boundaries."""
    token = re.compile(r"^\s*//")

    result = []
    for line in lines:
        if not token.match(line):
            result.append(line)

    return "".join(result)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("test_name", help="CTest test name or substring")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Project workspace root (auto-detected if omitted)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build directory (auto-detected if omitted)",
    )
    args = parser.parse_args()

    workspace = args.workspace or find_workspace_root()
    build_dir = args.build_dir or workspace / "build"

    test = find_test(build_dir, args.test_name)
    print(f"Found: {test['name']}", file=sys.stderr)

    config = make_config(test, workspace)

    launch_path = workspace / ".vscode" / "launch.json"
    if launch_path.exists():
        with open(launch_path, "r") as f:
            launch = json.loads(strip_jsonc_comments(f))
    else:
        launch_path.parent.mkdir(parents=True, exist_ok=True)
        launch = {"version": "0.2.0"}
    all_config = launch.get("configurations", [])
    launch["configurations"] = [config] + all_config  # type: ignore

    launch_path.write_text(json.dumps(launch, indent=4) + "\n")
    print(f"Updated {launch_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
