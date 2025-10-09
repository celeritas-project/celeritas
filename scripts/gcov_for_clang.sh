#!/bin/bash
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
#
# SYNOPSIS
#   gcov_for_clang.sh gcov-options
#
# DESCRIPTION
#   A wrapper script that provides gcov-compatible interface for LLVM's
#   llvm-cov tool when using Clang compiler. This script automatically
#   detects and uses the appropriate LLVM coverage tool version.
#
#   If `LLVM_COV` is set in the environment, it uses that as the coverage tool.
#   If not set, it first attempts to use 'llvm-cov' if available in PATH.
#   If not found, it determines the Clang version and looks for a
#   version-specific instance like 'llvm-cov-18'.
#
#   If `llvm-cov` is not found issues an error message and list available
#   `llvm-cov` found on the PATH.
#
# ENVIRONMENT VARIABLES
#   LLVM_COV  - Path to the LLVM coverage tool (optional)
#   CXX       - C++ compiler to detect version from (defaults to clang++)
#
# USAGE EXAMPLES
#   # Basic usage (gcovr will call this script)
#   gcovr --gcov-executable=./scripts/gcov_for_clang.sh
#
# EXIT STATUS
#   0   Success
#   1   LLVM coverage tool not found
#   *   Exit code from llvm-cov gcov command
#

if [ -z ${LLVM_COV+x} ];
then
  if command -v llvm-cov >/dev/null 2>&1; then
    LLVM_COV=$(command -v llvm-cov)
  else
    if [ -z ${CXX+x} ];
    then
      CXX=clang++
    fi
    CLANG_MAJOR=$(${CXX} --version | grep -oE 'clang version [0-9]+' | grep -oE '[0-9]+')
    LLVM_COV=llvm-cov-${CLANG_MAJOR}
  fi
fi

# Check if the LLVM_COV command exists
if ! command -v "${LLVM_COV}" >/dev/null 2>&1; then
  echo "Error: LLVM coverage tool '${LLVM_COV}' not found in PATH" >&2
  echo "Available llvm-cov variants:" >&2
  # Search for llvm-cov variants in PATH
  found_variants=false
  for path_dir in $(echo "$PATH" | tr ':' ' '); do
    if [ -d "$path_dir" ]; then
      for variant in "$path_dir"/llvm-cov*; do
        if [ -x "$variant" ]; then
          echo "  $(basename "$variant")" >&2
          found_variants=true
        fi
      done
    fi
  done
  if [ "$found_variants" = false ]; then
    echo "  No llvm-cov variants found" >&2
  fi
  exit 1
fi

exec ${LLVM_COV}  gcov "$@"
