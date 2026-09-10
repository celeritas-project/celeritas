#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run clang-tidy only on changed C++ files and report diagnostics on changed
# lines.
#-----------------------------------------------------------------------------#

set -e
log() {
  printf "%s: %s\n" "$1" "$2" >&2
}

BUILD_DIR="$PWD/build"
CLANG_TIDY_DIFF="$(dirname "$CLANG_TIDY")/../share/clang/clang-tidy-diff.py"
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

if [ ! -f "$CLANG_TIDY_DIFF" ]; then
  log error "clang-tidy-diff.py not found: $CLANG_TIDY_DIFF"
  exit 1
fi

log info "Fetching base commit ${BASE_SHA} from ${REMOTE}"
git fetch --depth 1 "${REMOTE}" "${BASE_SHA}"

log info "Using clang-tidy: $CLANG_TIDY"
git diff --diff-filter=ACM -U0 "$BASE_SHA"..."$HEAD_SHA" \
  | python3 "$CLANG_TIDY_DIFF" \
      -clang-tidy-binary "$CLANG_TIDY" \
      -p 1 \
      -path "$BUILD_DIR" \
      -extra-arg=-isysroot \
      -extra-arg=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
      -regex '^(src|app|test)/.*\.(cc|hh)$' \
      -only-check-in-db
