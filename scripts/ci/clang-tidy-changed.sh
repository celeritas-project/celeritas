#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

SOURCE_DIR="$PWD"
BUILD_DIR="$PWD/build"
BASE_REF="$1"

log() {
  printf "%s: %s\n" "$1" "$2" >&2
}
if [ -z "${BASE_REF}" ]; then
  log error "base ref not defined: pass as first argument"
  exit 1
fi

if [ -z "$CLANG_TIDY" ]; then
  log error "CLANG_TIDY not defined"
  exit 1
fi

log info "Finding changes from ${BASE_REF}"
BASE=$(git merge-base origin/${BASE_REF} HEAD)
ALL_FILES=$(git diff --name-only --diff-filter=ACM "$BASE" HEAD)
set +e
CC_FILES=$(grep -E '^(src|app)/.*\.cc$' - <<< "$ALL_FILES")

# Get list of files from compile_commands.json and filter CC_FILES
COMPILED_FILES=$(jq -r '.[].file' $BUILD_DIR/compile_commands.json)
CC_FILES=$(echo "$CC_FILES" | while read -r file; do
  if echo "$COMPILED_FILES" | grep -qE "^.*/${file}$"; then
    echo "$file"
  fi
done)
set -e
if [ -z "$CC_FILES" ]; then
  log info "No files to run clang-tidy on."
  exit 0
fi
log info "Running clang-tidy on: $CC_FILES"
$CLANG_TIDY -p $BUILD_DIR $CC_FILES
exit $?
