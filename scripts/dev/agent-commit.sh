#!/bin/sh -e
# Copyright 2026 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Commit staged and unstaged changes using a pre-written commit message file.
#
# Usage:
#   scripts/dev/agent-commit.sh <message-file> [--no-verify]
#
# Intended for use by AI coding agents that cannot reliably pass multi-line
# strings via shell heredocs. The agent writes the commit message to a file
# (e.g. build/commit_msg.txt), calls this script, and the file is deleted on
# success so that create_file can be used again next time.
#
# The commit is tagged with an Assisted-by trailer whose value is taken from
# the AGENT_TRAILER environment variable, defaulting to "GitHub Copilot".
#-----------------------------------------------------------------------------#

if [ $# -lt 1 ]; then
  printf "Usage: %s <message-file> [extra git-commit options...]\n" "$0" >&2
  exit 1
fi

MSG_FILE="$1"
shift

if [ ! -f "$MSG_FILE" ]; then
  printf "error: message file not found: %s\n" "$MSG_FILE" >&2
  exit 1
fi

TRAILER="${AGENT_TRAILER:-GitHub Copilot}"

# Stage all changes (including new files)
git add -A

# Run pre-commit hooks, then re-stage any files they modified
pre-commit run || git add -A

# Commit
git commit \
  --trailer "Assisted-by: ${TRAILER}" \
  -F "$MSG_FILE" \
  "$@"

# Clean up so the agent can use create_file next time
rm "$MSG_FILE"
