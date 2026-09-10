#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Optional

LAUNCH_ENV_UPDATES: dict[str, str] = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "CELER_LOG": "debug",
    "CELER_LOG_LOCAL": "debug",
    "CELER_LOG_SCOPED": "debug",
    "CELER_ENABLE_PROFILING": "0",
    "CELER_PROFILE_DEVICE": "0",
    "CELER_STRIP_SOURCEDIR": "1",
    "GTEST_COLOR": "1",
}

DRY_RUN = "dry-run"
TEST = "test"
REGRESSION = "regression"


class Harness:
    """Manage paths and run names for the test harness.

    - Actual files (failures or force-regen) are written to "$BUILD/regression/{subdir}/{name}"
      where subdir/name are based on the ctest name
    - Expected files live in "$SOURCE/test/{subdir}/regression/
    """

    def __init__(self, source_dir: Path, build_dir: Path):
        subdir, name = self._load_name()
        test_source_dir = source_dir / TEST / subdir

        self.name = name
        self.expected_dir = test_source_dir / REGRESSION
        self.actual_dir = build_dir / REGRESSION / subdir
        self.build_dir = build_dir

        if not self.expected_dir.parent.is_dir():
            print(f"error: test directory does not exist at {test_source_dir}")
            sys.exit(1)

    def _load_name(self) -> tuple[str, str]:
        try:
            name = os.environ["CELER_TEST_NAME"]
        except KeyError:
            name = f"temp/{DRY_RUN}:1"
            print(
                f"error: CELER_TEST_NAME is not set: using {DRY_RUN}", file=sys.stderr
            )

        # Strip "accel/", "app/foo/", etc
        subdir, _, name = name.rpartition("/")

        # Make CTest harness name safe for paths
        return subdir, name.replace(":", ".")

    def _make_path(self, root: Path, stream_name: str) -> Path:
        suffix = stream_name.replace("std", "")
        return root / f"{self.name}.{suffix}.txt"

    def make_expected_path(self, stream_name: str) -> Path:
        return self._make_path(self.expected_dir, stream_name)

    def make_actual_path(self, stream_name: str) -> Path:
        return self._make_path(self.actual_dir, stream_name)


class OutputReference:
    def __init__(self, name: str, actual_lines: list[str], harness: Harness):
        self.name = name
        self.actual_lines = actual_lines
        self.expected_lines: list[str] = []
        self.expected = harness.make_expected_path(name)
        self.actual = harness.make_actual_path(name)

        if self.expected.exists():
            self.expected_lines = self.expected.read_text().splitlines()


class Stream:
    def __init__(self, name: str, stream: IO[str], verbose: bool = True):
        self.name = name.replace("std", "")
        self.stream = stream
        self.output: list[str] = []
        self.verbose = verbose

    def capture_stream(self) -> None:
        prefix = self.name[0].upper() + ">"
        append = self.output.append
        verbose = self.verbose
        try:
            for i, line in enumerate(iter(self.stream.readline, "")):
                cleaned = normalize_line(line.rstrip())
                if verbose:
                    print(prefix, cleaned, file=sys.stderr)
                append(strip_ansi(cleaned))
                if not line:
                    # EOF
                    break
        finally:
            self.stream.close()


LINE_RE = re.compile(
    "("
    r"(?:\.?\.?/|/)?"  # leading dots or slash
    r"(?:[^\s:@]+/)*"  # optional interior components
    r"[^@\s:]+\.[^@\s:]+"  # final path component with extension
    ")"  # end path group
    r"(:\d+)?",  # optional line number
)


def normalize_line(line: str) -> str:
    """Remove run-dependent output from the line."""
    line = LINE_RE.sub(lambda match: Path(match.group(1)).name, line)
    line = re.sub(r"0x[a-f0-9]+", "0x0000", line)
    line = re.sub(r"\bversion \S+", "version [suppressed]", line)
    line = re.sub(
        r"\(\d+ ms(?: total)?\)$", "(time suppressed)", line
    )  # googletest time
    return line


def strip_ansi(line: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", line)


def make_run_command(
    exe: Path, args: list[str], child_env: dict[str, str]
) -> list[str]:
    """Construct a shell-like command with current environment variables and the executable+args.

    Note that it's assumed that most environment variable that affect execution will be output during the google test. This avoids the need to filter out CI- and spack-related environment.
    """
    lines = []
    include_keys = [key for key in child_env if key.startswith("G4_")]
    for key in sorted(include_keys):
        val = shlex.quote(normalize_line(child_env[key]))
        lines.append(f"{key}={val} \\")
    lines.append(" ".join(shlex.quote(a) for a in [str(exe)] + args))
    return lines


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(s + "\n" for s in lines)


def print_diff(output: OutputReference) -> None:
    diff = difflib.unified_diff(
        output.expected_lines,
        output.actual_lines,
        fromfile=str(output.expected),
        tofile=str(output.actual),
        lineterm="",
    )
    for line in diff:
        sys.stdout.write(line + "\n")


def compare_output(output: OutputReference, force_regen: bool) -> bool:
    success = output.expected_lines == output.actual_lines

    what = "matching"
    if success:
        print(f"{output.name} matches {output.expected} contents")
    elif output.expected_lines:
        what = "failed"
        # Expected output exists but is different
        print(f"FAILED: diff for {output.name}:")
        print_diff(output)
    else:
        what = "missing"

    if force_regen or not success:
        print(f"Writing {what} output to: {output.actual}")
        write_text(output.actual, output.actual_lines)

    if (force_regen and not success) or not output.expected_lines:
        write_text(output.expected, output.actual_lines)
        print(f"Writing {what} output to source dir: {output.expected}")

    return success


def run(
    harness: Harness,
    exe: Path,
    args: list[str],
    *,
    quiet: bool,
    force_regen: bool,
    timeout: Optional[float],
):
    child_env = dict(os.environ)
    child_env.update(LAUNCH_ENV_UPDATES)
    cmd_text = make_run_command(exe.relative_to(harness.build_dir), args, child_env)

    # For improved reproducibility (executables that echo $0 and don't have a path suffix)
    # launch with the relative directory
    exe = exe.relative_to(Path.cwd())
    print(
        "Running",
        shlex.quote(str(exe)),
        *(shlex.quote(a) for a in args),
        file=sys.stderr,
    )
    process = subprocess.Popen(
        ["./" + str(exe)] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=128,
        env=child_env,
    )

    streams = {
        k: Stream(k, getattr(process, k), verbose=not quiet)
        for k in ["stdout", "stderr"]
    }
    threads = [threading.Thread(target=s.capture_stream) for s in streams.values()]
    for t in threads:
        t.start()
    returncode = process.wait(timeout=timeout)
    for t in threads:
        t.join()
    if returncode:
        print(f"error: {exe} returned {returncode}")

    if not harness.actual_dir.exists():
        print("Creating parent directory for actual output")
        harness.actual_dir.mkdir(parents=True)

    outputs = [
        OutputReference(name=k, actual_lines=cmd_text + s.output, harness=harness)
        for k, s in streams.items()
    ]
    success = True
    for output in outputs:
        success = compare_output(output, force_regen=force_regen) and success
    if not success and returncode == 0:
        returncode = 1
    return returncode


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an executable's stdout/stderr against expected outputs.",
        epilog="Use '--' before command to allow other dashed arguments",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=f"project source directory (expected results live in source/test/subdir/{REGRESSION}",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help=f"project build directory (failed/newly generated output lives in build/{REGRESSION}/subdir",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not echo stdout/err to console during run",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="rewrite the expected output files instead of comparing them",
    )
    parser.add_argument(
        "--timeout",
        nargs="?",
        type=float,
        default=None,
        help="stop the process after a maximum time",
    )
    parser.add_argument(
        "command",
        nargs="+",
        help="the executable to run followed by any arguments",
    )

    args = parser.parse_args(argv)
    if args.build_dir is None:
        print(
            "warning: --build-dir not set; defaulting to current working directory",
            file=sys.stderr,
        )
        args.build_dir = Path.cwd()
    if args.source_dir is None:
        print(
            "warning: --source-dir not set; defaulting to great-grandparent directory",
            file=sys.stderr,
        )
        args.source_dir = Path(__file__).resolve().parents[2]

    for k in ["source_dir", "build_dir"]:
        d: Path = getattr(args, k)
        if not d.is_dir():
            parser.error(f"{k} is not a directory: {d}")

    executable = Path(args.command[0])
    if not executable.exists():
        parser.error(f"executable does not exist: {executable}")
    if executable.is_dir() or not os.access(executable, os.X_OK):
        parser.error(f"not an executable file: {executable}")

    harness = Harness(source_dir=args.source_dir, build_dir=args.build_dir)

    return run(
        harness,
        executable,
        args.command[1:],
        quiet=args.quiet,
        force_regen=args.force_regen,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
