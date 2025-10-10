#!/bin/sh
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
#   llvm-cov tool when using Clang compiler.

#   If `LLVM_COV` is set in the environment, it uses that as the coverage tool.
#   If not set, attempts to use 'llvm-cov' if available in PATH.
#
#   If `llvm-cov` is not found issues an error message.
#
# ENVIRONMENT VARIABLES
#   LLVM_COV  - Path to the LLVM coverage tool (optional)
#
# USAGE EXAMPLES
#   # Basic usage (gcovr will call this script)
#   gcovr --gcov-executable=./scripts/gcov_for_clang.sh ...
#
# EXIT STATUS
#   0   Success
#   1   LLVM coverage tool not found
#   *   Exit code from llvm-cov gcov command
#

if [ -z "${LLVM_COV}" ];
then
  if command -v llvm-cov >/dev/null 2>&1; then
    LLVM_COV=$(command -v llvm-cov)
  else
    echo "Error: LLVM_COV is not set and LLVM coverage tool 'llvm-cov' not found in PATH" >&2
    exit 1
  fi
else
  if ! command -v "${LLVM_COV}" >/dev/null 2>&1; then
    echo "Error: LLVM coverage tool '${LLVM_COV}' not found in PATH" >&2
    exit 1
  fi
fi

exec "${LLVM_COV}" gcov "$@"
