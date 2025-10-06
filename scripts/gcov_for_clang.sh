#!/bin/bash

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
