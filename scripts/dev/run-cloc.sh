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

cd $SOURCE_DIR
echo "Source/utility code:"
cloc --git HEAD --fullpath --not-match-d='/test/' $@
echo "Test code:"
cloc --git HEAD --fullpath --match-d='/test/' $@
