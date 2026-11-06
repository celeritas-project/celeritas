# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# Shared helpers for eval scaffold scripts. Source this file after setting
# CASE_DIR to the case directory:
#
#   CASE_DIR=$(cd "$(dirname "$0")" && pwd)
#   . "$CASE_DIR/../_lib/fixtures.sh"
#
# Scaffolds run in the empty eval workspace with only PATH from the host
# environment. The Celeritas source tree is found relative to the case
# directory (contrib/claude-plugin/evals/<case>).

REPO_ROOT=$(cd "$CASE_DIR/../../../.." && pwd)
G4_EXAMPLE="$REPO_ROOT/example/geant4"

require_repo() {
  if [ ! -f "$G4_EXAMPLE/add-celer.diff" ]; then
    echo "error: these evals need a Celeritas source checkout" \
      "(no $G4_EXAMPLE/add-celer.diff)" >&2
    exit 1
  fi
}

# Copy the Geant4 offload examples and their CMake file into the workspace
copy_geant4_example() {
  require_repo
  cp "$G4_EXAMPLE"/*.cc "$G4_EXAMPLE/CMakeLists.txt" .
}

# Same, with the Celeritas offload changes removed (as in
# scripts/ci/test-examples.sh)
copy_geant4_example_without_offload() {
  copy_geant4_example
  patch -s -p2 -R < "$G4_EXAMPLE/add-celer.diff"
}
