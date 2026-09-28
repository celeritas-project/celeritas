#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Check out the Spack and spack-packages revisions pinned by the setup-spack
# CI action of the Celeritas source at CELER_SOURCE_DIR, then run a command.
#-----------------------------------------------------------------------------#

set -e
log() {
  printf "%s: %s\n" "$1" "$2" >&2
}

ACTION_FILE="${CELER_SOURCE_DIR}/.github/actions/setup-spack/action.yml"
if [ ! -f "${ACTION_FILE}" ]; then
  log error "${ACTION_FILE} not found: mount a Celeritas checkout at ${CELER_SOURCE_DIR}"
  exit 1
fi

# Print "<destination> <url> <sha>" for each Spack checkout in the action
refs=$(python3 - "${ACTION_FILE}" <<'EOF'
import os
import sys

import yaml

dests = {
    "spack/spack": os.environ["SPACK_ROOT"],
    "spack/spack-packages": os.environ["SPACK_PACKAGES_REPO"],
}
with open(sys.argv[1]) as f:
    steps = yaml.safe_load(f)["runs"]["steps"]
for step in steps:
    inputs = step.get("with") or {}
    repo = inputs.get("repository")
    if repo in dests:
        print(dests.pop(repo), f"https://github.com/{repo}.git", inputs["ref"])
if dests:
    sys.exit("error: no pinned ref for " + ", ".join(dests))
EOF
)

printf "%s\n" "${refs}" | while read -r dest url sha; do
  if [ ! -d "${dest}/.git" ]; then
    git init -q "${dest}"
    git -C "${dest}" remote add origin "${url}"
  fi
  if [ "$(git -C "${dest}" rev-parse -q --verify HEAD || true)" != "${sha}" ]; then
    log info "Checking out ${url} at ${sha}"
    git -C "${dest}" fetch -q --depth 1 origin "${sha}"
    git -C "${dest}" checkout -q --detach FETCH_HEAD
  fi
done

# Use the pinned package repository instead of letting Spack clone its own
spack repo set --destination "${SPACK_PACKAGES_REPO}" builtin

exec "$@"
