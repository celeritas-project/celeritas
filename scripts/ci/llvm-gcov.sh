#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
#
# SYNOPSIS
#   llvm-gcov.sh gcov-options
#
# DESCRIPTION
#   A wrapper script that provides gcov-compatible interface for LLVM's
#   llvm-cov tool when using Clang compiler.
#
# ENVIRONMENT VARIABLES
#   LLVM_COV  - Path to the LLVM coverage tool (optional)
#
# USAGE EXAMPLES
#   # Basic usage (gcovr will call this script)
#   gcovr --gcov-executable=./scripts/ci/llvm-gcov.sh ...
#

if [ -z "${LLVM_COV}" ];
then
  LLVM_COV=llvm-cov
  echo "Using default LLVM_COV=${LLVM_COV}" >&2
fi

if ! command -v "${LLVM_COV}" >/dev/null 2>&1; then
  echo "Error: LLVM coverage tool '${LLVM_COV}' not found in PATH" >&2
  exit 1
fi

exec "${LLVM_COV}" gcov "$@"
