#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# SessionStart hook (registered in .claude/settings.json) for Claude Code
# cloud sessions, which start from a fresh shallow clone without tags: fetch
# the full history and tags (a few seconds) so that CMake's 'git describe'
# finds the latest version tag. This does nothing outside cloud sessions or
# when the clone is complete and has a version tag.
#-----------------------------------------------------------------------------#

set -e

if [ "${CLAUDE_CODE_REMOTE}" != "true" ]; then
  exit 0
fi
cd "${CLAUDE_PROJECT_DIR:-.}"

unshallow=
if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
  unshallow=--unshallow
elif git describe --tags --match 'v*' >/dev/null 2>&1; then
  exit 0
fi

n=1
until git fetch -q ${unshallow} --tags; do
  if [ "${n}" -ge 3 ]; then
    echo "Failed to fetch git history and tags: the Celeritas version will be unknown" >&2
    exit 1
  fi
  sleep $((n * 2))
  n=$((n + 1))
done

version="$(git describe --tags --match 'v*' 2>/dev/null)" \
  || version="no version tag found"
echo "Fetched full git history and tags: ${version}"
