#!/bin/sh -e
BUILD_SUBDIR=build
exec "$(cd "$(dirname $BASH_SOURCE[0])" && pwd)/emmet.sh"
