#!/bin/sh -e
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
exec "${BUILDSCRIPT_DIR}/emmet.sh"
