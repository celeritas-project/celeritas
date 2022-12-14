#!/bin/sh -ex
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
OUTFILE=$(mktemp)
iwyu_tool.py -p $1 -- \
  -Xiwyu --no_fwd_decls \
  -Xiwyu --keep="*device_runtime_api*" \
  -Xiwyu --keep="*.json.hh*" \
  -Xiwyu --transitive_includes_only \
  -Xiwyu --mapping_file="${SCRIPT_DIR}/iwyu-apple-clang.imp" \
  > $OUTFILE \
|| echo "error: iwyu failed"


SKIP_FORMAT=
if ! (cd "${SCRIPT_DIR}" && git diff-files --quiet) ; then
  echo "warning: Git repository is dirty so patches will not be applied"
  SKIP_FORMAT=true
else
  fix_includes.py --nocomments -p $1 < $OUTFILE
fi

if [ -z "${SKIP_FORMAT}" ]; then
  # Fix include ordering
  git add -u :/
  SKIP_GCF=1 git commit -m "IWYU" --author="Mr. Clean <noreply@github.com>"
  git-clang-format HEAD^
  git add -u :/
  SKIP_GCF=1 git commit --amend -m "IWYU+Clean" >/dev/null
  git reset HEAD^
fi
