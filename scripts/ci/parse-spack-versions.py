#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Print the Spack revisions pinned by the setup-spack CI action.

The output is a pair of shell assignments suitable for ``eval``::

    SPACK_REF=<sha>
    SPACK_PACKAGES_REF=<sha>

It is used to build and check the Spack buildcache container image (see
``scripts/docker/buildcache``). Only the standard library is used so that this
runs with any host python3.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

VARIABLES = {
    "spack/spack": "SPACK_REF",
    "spack/spack-packages": "SPACK_PACKAGES_REF",
}
STEP_RE = re.compile(r"^\s*-\s")
REPO_RE = re.compile(r"^\s*repository:\s*(\S+)")
REF_RE = re.compile(r"^\s*ref:\s*(\S+)")
SHA_RE = re.compile(r"[0-9a-f]{40}")


def parse_pins(lines):
    """Map each pinned Spack repository to the ref of its checkout step."""
    pins = {}
    repo = None
    ref = None
    # Append a sentinel step to close the last one
    for line in lines + ["- "]:
        repo_match = REPO_RE.match(line)
        ref_match = REF_RE.match(line)
        if STEP_RE.match(line):
            if repo in VARIABLES and ref is not None:
                pins[repo] = ref
            repo = ref = None
        elif repo_match:
            repo = repo_match.group(1)
        elif ref_match:
            ref = ref_match.group(1)
    return pins


def default_action_path():
    """Locate the setup-spack action in the enclosing git repository."""
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit("error: not in a git repository; specify the action path")
    path = Path(root) / ".github" / "actions" / "setup-spack" / "action.yml"
    if not path.is_file():
        sys.exit(f"error: {path} does not exist; specify the action path")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "action",
        nargs="?",
        type=Path,
        help="path to setup-spack/action.yml (default: from git repo root)",
    )
    args = parser.parse_args()
    path = args.action or default_action_path()
    with open(path) as f:
        pins = parse_pins(f.read().splitlines())

    missing = [repo for repo in VARIABLES if repo not in pins]
    if missing:
        sys.exit("error: no pinned ref for " + ", ".join(missing))
    for repo, var in VARIABLES.items():
        if not SHA_RE.fullmatch(pins[repo]):
            sys.exit(f"error: ref for {repo} is not a commit SHA: {pins[repo]}")
        print(f"{var}={pins[repo]}")


if __name__ == "__main__":
    main()
