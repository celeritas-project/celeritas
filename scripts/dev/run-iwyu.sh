#!/bin/sh -ex
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# NOTE: this is recommended to be run on a build without debug assertions,
# tests, or demos.

if ! hash include-what-you-use ; then
  echo "This script requires https://github.com/include-what-you-use/include-what-you-use"
  exit 1
fi

if [ $# -ne 1 ]; then
  echo "Usage: $0 compile_commands.json" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname $0)" && pwd)"
if ! (cd "${SCRIPT_DIR}" && git diff-files --quiet) ; then
  echo "$0: error: will not run on a dirty Git repository"
  exit 0
fi

OUTFILE=$(mktemp)
iwyu_tool.py -p $1 -- \
  -Xiwyu --no_fwd_decls \
  -Xiwyu --keep="*device_runtime_api*" \
  -Xiwyu --keep="*.json.hh*" \
  -Xiwyu --transitive_includes_only \
  -Xiwyu --mapping_file="${SCRIPT_DIR}/iwyu-apple-clang.imp" \
  | grep -v '\.icc>' \
  > $OUTFILE \
|| echo "error: iwyu failed"

fix_includes.py --nocomments -p $1 < $OUTFILE

if [ -z "${SKIP_FORMAT}" ]; then
  # Fix include ordering
  git add -u :/
  SKIP_GCF=1 git commit -m "IWYU" --author="Mr. Clean <noreply@github.com>" >/dev/null
  git-clang-format HEAD^ || true
  git add -u :/
  SKIP_GCF=1 git commit --amend -m "IWYU+Clean" >/dev/null
  git reset HEAD^
  git co HEAD -- ":/src/celeritas/*/generated/*"
fi
