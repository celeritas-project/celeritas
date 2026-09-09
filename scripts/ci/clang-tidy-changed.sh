#!/bin/bash
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# TODO: replace this script with something that parses header dependencies
# with clang and determines what .cc files must be compiled to test *all*
# the changes, including to headers, in the src/ and app/ directories.
# (Currently the files in test/ have too many issues.)
#-----------------------------------------------------------------------------#

set -e
log() {
  printf "%s: %s\n" "$1" "$2" >&2
}

BUILD_DIR="$PWD/build"
REMOTE="$1"
BASE_SHA="$2"
HEAD_SHA="HEAD"

if [ $# -ne 2 ]; then
  log usage "CLANG_TIDY=path $0 remote base_sha"
  exit 1
fi

if [ -z "$CLANG_TIDY" ]; then
  log error "CLANG_TIDY not defined"
  exit 1
fi

log info "Fetching base commit ${BASE_SHA} from ${REMOTE}"
git fetch --depth 1 "${REMOTE}" "${BASE_SHA}"

# NOTE: this only compares source/app code files that have changed, and does
# not process changes to headers.
ALL_FILES=$(git diff --name-only --diff-filter=ACM "$BASE_SHA"..."$HEAD_SHA")
CC_FILES=$(grep -E '^(src|app)/.*\.cc$' - <<< "$ALL_FILES") || {
  log info "No *.cc files have changed."
  exit 0
}

# Get list of files from compile_commands.json and filter CC_FILES
# (NOTE: this is O(N^2) for large commits: maybe this script should use python
# and also fix the fact that .hh files are not checked)
COMPILED_FILES=$(jq -r '.[].file' "$BUILD_DIR/compile_commands.json")
CC_FILES=$(echo "$CC_FILES" | while read -r file; do
  if echo "$COMPILED_FILES" | grep -qE "^.*/${file}$"; then
    echo "$file"
  fi
done)
if [ -z "$CC_FILES" ]; then
  log info "No files to run clang-tidy on."
  exit 0
fi
log info "Running clang-tidy on: $CC_FILES"
$CLANG_TIDY -p $BUILD_DIR $CC_FILES
