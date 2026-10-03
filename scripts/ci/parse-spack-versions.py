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

import re
import sys

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


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} path/to/setup-spack/action.yml")
    with open(sys.argv[1]) as f:
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
