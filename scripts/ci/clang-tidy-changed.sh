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
REMOTE="$1"
BASE_SHA="$2"
HEAD_SHA="HEAD"

if [ $# -ne 2 ]; then
  log usage "CLANG_TIDY=path CLANG_TIDY_DIFF=otherpath $0 remote base_sha"
  exit 1
fi

if [ -z "$CLANG_TIDY" ]; then
  log error "CLANG_TIDY not defined"
  exit 1
fi

if [ -z "$CLANG_TIDY_DIFF" ]; then
  log error "CLANG_TIDY_DIFF not defined"
  exit 1
fi

if [ ! -f "$CLANG_TIDY_DIFF" ]; then
  log error "clang-tidy-diff.py not found: $CLANG_TIDY_DIFF"
  exit 1
fi

log info "Fetching base commit ${BASE_SHA} from ${REMOTE}"
git fetch --depth 1 "${REMOTE}" "${BASE_SHA}"

log info "Using clang-tidy: $CLANG_TIDY"
# TODO: Remove the warning ignore when upgrading to LLVM 20 or newer, whose driver has
# invalid escapes.
diff_file=$(mktemp)
tidy_status_file=$(mktemp)
trap 'rm -f "$diff_file" "$tidy_status_file"' 0

git diff --diff-filter=ACM -U0 "$BASE_SHA"..."$HEAD_SHA" > "$diff_file"

if grep -qE '^\+\+\+ b/(src|app|test)/.*\.hh$' "$diff_file"; then
  log info "Header changed: running clang-tidy on all compiled sources"
  (
    set +e
    run-clang-tidy -clang-tidy-binary "$CLANG_TIDY" -p "$BUILD_DIR" 2>&1
    tidy_status=$?
    printf '%s\n' "$tidy_status" > "$tidy_status_file"
  ) | awk '
    function escape_annotation(value) {
      gsub(/%/, "%25", value)
      gsub(/\r/, "%0D", value)
      gsub(/\n/, "%0A", value)
      return value
    }

    {
      print
    }

    /^[^:]+:[0-9]+:[0-9]+: error: / {
      split($0, diagnostic, ":")
      message = $0
      sub(/^[^:]+:[0-9]+:[0-9]+: error: /, "", message)
      print "========== CLANG-TIDY ERROR =========="
      print "::error file=" diagnostic[1] ",line=" diagnostic[2] ",col=" diagnostic[3] "::" escape_annotation(message)
      print "======================================="
    }
  '
  tidy_status=$(cat "$tidy_status_file")
  if [ "$tidy_status" -ne 0 ]; then
    exit "$tidy_status"
  fi
else
  python3 -W ignore::SyntaxWarning "$CLANG_TIDY_DIFF" \
    -clang-tidy-binary "$CLANG_TIDY" \
    -p 1 \
    -path "$BUILD_DIR" \
    -regex '^(src|app|test)/.*\.cc$' \
    < "$diff_file"
fi
