#!/bin/sh -e
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# Geant4 example with two integration bugs. SetOptions moves after
# Initialize(), which a careful reader can spot. The SD also scores energy in a
# vector indexed by track ID, which is only a bug if you know that tracks
# created by Celeritas have IDs counting down from INT_MAX.
CASE_DIR=$(cd "$(dirname "$0")" && pwd)
. "$CASE_DIR/../_lib/fixtures.sh"
copy_geant4_example

# The other examples would give away the correct setup order
rm -f simple-offload.cc fastsim-offload.cc
cat > CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.18...4.1)
project(CeleritasAccelExample VERSION 0.1.0 LANGUAGES CXX)

find_package(Celeritas 0.7 REQUIRED)
find_package(Geant4 REQUIRED)

add_executable(trackingmanager-offload trackingmanager-offload.cc)
celeritas_target_link_libraries(trackingmanager-offload
  Celeritas::G4
  ${Geant4_LIBRARIES}
)
EOF

src=trackingmanager-offload.cc
awk '
  /tmi\.SetOptions\(MakeOptions\(\)\);/ { held = $0; next }
  /^#include <type_traits>/ { print; print "#include <vector>"; next }
  /void Initialize\(G4HCofThisEvent\*\) final \{ edep_ = 0; \}/ {
    print "    void Initialize(G4HCofThisEvent*) final"
    print "    {"
    print "        edep_ = 0;"
    print "        edep_by_track_.clear();"
    print "    }"
    next
  }
  /edep_ \+= step->GetTotalEnergyDeposit\(\);/ {
    print
    print ""
    print "        // Score energy per track for the truth record"
    print "        auto const index"
    print "            = static_cast<std::size_t>(step->GetTrack()->GetTrackID());"
    print "        if (index >= edep_by_track_.size())"
    print "        {"
    print "            edep_by_track_.resize(index + 1);"
    print "        }"
    print "        edep_by_track_[index] += step->GetTotalEnergyDeposit();"
    next
  }
  /double edep_\{0\};/ { print; print "    std::vector<double> edep_by_track_;"; next }
  { print }
  /run_manager->Initialize\(\);/ && held != "" { print held; held = "" }
' "$src" > "$src.tmp"
mv "$src.tmp" "$src"

# Make sure the bugs were actually injected
if ! awk '/->Initialize\(\);/ { init = NR } /tmi\.SetOptions/ { opt = NR }
          END { exit !(init && opt && opt > init) }' "$src"; then
  echo "error: failed to move SetOptions after Initialize in $src" >&2
  exit 1
fi
for injected in '#include <vector>' 'edep_by_track_.clear();' \
    'edep_by_track_.resize(index + 1);' 'std::vector<double> edep_by_track_;'; do
  if ! grep -qF "$injected" "$src"; then
    echo "error: failed to inject '$injected' into $src" >&2
    exit 1
  fi
done
