#!/bin/sh -e
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
###############################################################################
# Examples:
#   run-cloc.sh --by-file
#   run-cloc.sh --csv
###############################################################################
SCRIPT_DIR="$(cd "$(dirname $0)" && pwd)"
SOURCE_DIR="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
COMMIT=HEAD

if ! hash cloc ; then
  echo "This script requires https://github.com/AlDanial/cloc"
  exit 1
fi

function comment() {
  printf "\e[2;37;40m%s:\e[0m\n" "$1" >&2
}
function run_cloc() {
  cloc --git ${COMMIT} --force-lang=CUDA,hip $@
}

cd $SOURCE_DIR
comment "Library code"
run_cloc --exclude-dir=app,example,scripts,test $@
comment "App code"
run_cloc --fullpath --match-d='/app/' $@
comment "Example code"
run_cloc --fullpath --match-d='/example/' $@
comment "Test code"
run_cloc --fullpath --match-d='/test/' $@
