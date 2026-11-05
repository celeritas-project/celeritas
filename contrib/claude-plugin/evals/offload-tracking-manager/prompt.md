---
description: Add tracking-manager offload to the Geant4 example with offload removed
tags: [fixture, offload]
runs: 2
max_turns: 25
allowed_tools: [Read, Glob, Grep, Skill, Edit]
expected_outcome: Changes equivalent to example/geant4/add-celer.diff
---

This directory contains a Geant4 11.2 application,
`trackingmanager-offload.cc`, built by `CMakeLists.txt` against an installed
Celeritas 0.7. Add Celeritas offloading of electromagnetic tracks to it using
the recommended integration. Edit the files directly. Do not try to build or
run anything.
