#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Print the Spack revisions pinned by the setup-spack CI action.

The output is shell assignments suitable for ``eval``::

    SPACK_REPO=spack/spack
    SPACK_REF=<sha>
    SPACK_PACKAGES_REPO=spack/spack-packages
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
    "checkout-spack": ("SPACK_REPO", "SPACK_REF"),
    "checkout-spack-pkg": ("SPACK_PACKAGES_REPO", "SPACK_PACKAGES_REF"),
}
STEP_RE = re.compile(r"^\s*-\s")
ID_RE = re.compile(r"^\s*id:\s*(\S+)")
REPO_RE = re.compile(r"^\s*repository:\s*(\S+)")
REF_RE = re.compile(r"^\s*ref:\s*(\S+)")
SHA_RE = re.compile(r"[0-9a-f]{40}")


def parse_pins(lines):
    """Map each relevant checkout step ID to its repository and ref."""
    pins = {}
    step_id = None
    repo = None
    ref = None
    # Append a sentinel step to close the last one
    for line in lines + ["- "]:
        id_match = ID_RE.match(line)
        repo_match = REPO_RE.match(line)
        ref_match = REF_RE.match(line)
        if STEP_RE.match(line):
            if step_id in VARIABLES and repo is not None and ref is not None:
                pins[step_id] = (repo, ref)
            step_id = repo = ref = None
        elif id_match:
            step_id = id_match.group(1)
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

    missing = [step_id for step_id in VARIABLES if step_id not in pins]
    if missing:
        sys.exit("error: no pinned checkout step for " + ", ".join(missing))
    for step_id, (repo_var, ref_var) in VARIABLES.items():
        repo, ref = pins[step_id]
        if not SHA_RE.fullmatch(ref):
            sys.exit(f"error: ref for {step_id} is not a commit SHA: {ref}")
        print(f"{repo_var}={repo}")
        print(f"{ref_var}={ref}")


if __name__ == "__main__":
    main()
