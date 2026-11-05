---
description: Add offload to the Geant4 example, then build and run it against the host install
tags: [build]
runs: 1
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Edit, Write, Bash]
expected_outcome: The example builds against the installed Celeritas and its own energy check passes
---

This directory contains a Geant4 application, `trackingmanager-offload.cc`,
and its `CMakeLists.txt`. Add Celeritas offloading of electromagnetic tracks
to it using the recommended integration.

Then build and run it:
1. Run `. ./build-env.sh` in each shell command that needs the installed
   Celeritas and Geant4.
2. Configure with CMake into `./build` and build only the
   `trackingmanager-offload` target.
3. Run `./build/trackingmanager-offload` with `CELER_LOG=info`. It exits
   nonzero if its own check of the simulated energy fails.

Finally, write `RESULT.txt`. Its first line must be exactly `PASS` if the
program built and exited with status 0, and `FAIL` otherwise. Follow it with
the last 20 lines of the program's output.
