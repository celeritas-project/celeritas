#!/bin/sh -e
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# Geant4 example with offload, but SetOptions moved after Initialize()
CASE_DIR=$(cd "$(dirname "$0")" && pwd)
. "$CASE_DIR/../_lib/fixtures.sh"
copy_geant4_example
rm -f simple-offload.cc fastsim-offload.cc

src=trackingmanager-offload.cc
awk '
  /tmi\.SetOptions\(MakeOptions\(\)\);/ { held = $0; next }
  { print }
  /run_manager->Initialize\(\);/ && held != "" { print held; held = "" }
' "$src" > "$src.tmp"
mv "$src.tmp" "$src"

# Make sure the bug was actually injected
if ! awk '/->Initialize\(\);/ { init = NR } /tmi\.SetOptions/ { opt = NR }
          END { exit !(init && opt && opt > init) }' "$src"; then
  echo "error: failed to move SetOptions after Initialize in $src" >&2
  exit 1
fi
