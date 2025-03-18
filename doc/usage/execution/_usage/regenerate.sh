#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# Define the target directory
TARGET_DIR="$(cd "$(dirname "$0")" && pwd)"

cd bin || (
  echo "Run from the top level of the build directory"
  exit 1
)

# Iterate over each executable in the working directory
for app in *; do
    if [ -x "$app" ] && [ ! -d "$app" ]; then
        "./$app" --help > "$TARGET_DIR/$app.txt" 2>&1 
    fi
done
