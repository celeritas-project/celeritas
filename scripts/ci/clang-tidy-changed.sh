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

if ! CLANG_TIDY_PATH=$(command -v "$CLANG_TIDY"); then
  log error "clang-tidy not found: $CLANG_TIDY"
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
header_file=$(mktemp)
dependency_file=$(mktemp)
source_regex_file=$(mktemp)
trap 'rm -f "$diff_file" "$tidy_status_file" "$header_file" "$dependency_file" "$source_regex_file"' 0

git diff --diff-filter=ACM -U0 "$BASE_SHA"..."$HEAD_SHA" > "$diff_file"

if grep -qE '^\+\+\+ b/(src|app|test)/.*\.hh$' "$diff_file"; then
  awk '/^\+\+\+ b\/(src|app|test)\/.*\.hh$/ { sub(/^\+\+\+ b\//, ""); print }' "$diff_file" > "$header_file"
  tidy_directory=${CLANG_TIDY_PATH%/*}
  tidy_name=${CLANG_TIDY_PATH##*/}
  case "$tidy_name" in
    clang-tidy-*)
      scanner_suffix=${tidy_name#clang-tidy}
      ;;
    clang-tidy)
      scanner_suffix=
      ;;
    *)
      scanner_suffix=-18
      ;;
  esac
  CLANG_SCAN_DEPS=${CLANG_SCAN_DEPS:-$tidy_directory/clang-scan-deps$scanner_suffix}
  RUN_CLANG_TIDY=${RUN_CLANG_TIDY:-run-clang-tidy}

  if ! command -v "$CLANG_SCAN_DEPS" >/dev/null 2>&1; then
    log error "clang dependency scanner not found: $CLANG_SCAN_DEPS"
    exit 1
  fi
  if ! command -v "$RUN_CLANG_TIDY" >/dev/null 2>&1; then
    log error "run-clang-tidy not found: $RUN_CLANG_TIDY"
    exit 1
  fi

  log info "Header changes detected: finding affected source files"
  "$CLANG_SCAN_DEPS" \
    -compilation-database "$BUILD_DIR/compile_commands.json" \
    -format experimental-full \
    -o "$dependency_file"

    selected_count=$(python3 scripts/ci/clang-tidy-affected-sources.py \
    "$header_file" "$dependency_file" "$source_regex_file")

  if [ "$selected_count" -eq 0 ]; then
    log info "No compiled source file includes the changed headers"
    exit 0
  fi
  log info "Running clang-tidy on $selected_count affected source files"
  (
    set +e
    "$RUN_CLANG_TIDY" \
      -clang-tidy-binary "$CLANG_TIDY" \
      -p "$BUILD_DIR" \
      "$(cat "$source_regex_file")" 2>&1
    tidy_status=$?
    printf '%s\n' "$tidy_status" > "$tidy_status_file"
  ) | awk '
    function escape_annotation(value) {
      gsub(/%/, "%25", value)
      gsub(/\r/, "%0D", value)
      gsub(/\n/, "%0A", value)
      return value
    }

    /^[0-9]+ warnings generated\.$/ {
      generated[$1] = 1
      next
    }

    /^Suppressed [0-9]+ warnings \([0-9]+ in / {
      split($0, fields, " ")
      generated_count = fields[4]
      sub(/^\(/, "", generated_count)
      if (generated[generated_count]) {
        delete generated[generated_count]
        sub(/^Suppressed /, generated_count " warnings generated; ")
        sub(/ warnings \(/, " suppressed (", $0)
      }
      print
      next
    }

    /^Use -header-filter=\.\* to display errors from all non-system headers\./ {
      next
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
