#!/bin/sh -e
###############################################################################
# Examples:
#   run-cloc.sh --by-file
#   run-cloc.sh --csv
###############################################################################
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"

if ! hash cloc ; then
  echo "This script requires https://github.com/AlDanial/cloc"
  exit 1
fi

function run_cloc() {
  cloc --git HEAD --fullpath --force-lang=CUDA,hip $@
}

cd $SOURCE_DIR
echo "Source/utility code:"
run_cloc --not-match-d='/generated/'  --not-match-d='/test/' $@
echo "Test code:"
run_cloc --match-d='/test/' $@
