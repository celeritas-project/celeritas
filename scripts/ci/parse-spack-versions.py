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

REPO_PREFIX = {
    "checkout-spack": "SPACK",
    "checkout-spack-pkg": "SPACK_PACKAGES",
}
STEP_RE = re.compile(r"^\s*-\s")
KEY_RE = re.compile(r"^\s*(id|repository|ref):\s*(\S+)")
SHA_RE = re.compile(r"[0-9a-f]{40}")


def parse_pins(lines):
    """Map each relevant checkout step ID to its repository and ref."""
    pins = {}
    keys = {}
    for line in lines:
        # A new step processes any keys gathered
        if keys and STEP_RE.match(line):
            step_id = keys.pop("id")
            if step_id in REPO_PREFIX:
                pins[step_id] = (keys.get("repository"), keys.get("ref"))
            keys.clear()
            continue

        # Process key (presumably inside a step)
        key_match = KEY_RE.match(line)
        if key_match:
            keys[key_match.group(1)] = key_match.group(2)
    assert not keys
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

    missing = [step_id for step_id in REPO_PREFIX if step_id not in pins]
    if missing:
        sys.exit("error: no pinned checkout step for " + ", ".join(missing))
    for step_id, prefix in REPO_PREFIX.items():
        repo, ref = pins[step_id]
        if not SHA_RE.fullmatch(ref):
            sys.exit(f"error: ref for {step_id} is not a commit SHA: {ref}")
        print(f"{prefix}_REPO={repo}")
        print(f"{prefix}_REF={ref}")


if __name__ == "__main__":
    main()
