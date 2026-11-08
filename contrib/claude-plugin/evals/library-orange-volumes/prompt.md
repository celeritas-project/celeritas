---
description: Small C++ program using ORANGE geometry from an installed Celeritas
tags: [smoke, library]
runs: 2
max_turns: 20
allowed_tools: [Read, Glob, Grep, Skill]
---

Using an installed Celeritas 0.7 (built with Geant4) as a C++ library, write
a minimal program `main.cc` and its `CMakeLists.txt` that load
`detector.gdml` into Celeritas' ORANGE geometry and print the number of
volumes. Do not use the Geant4 offload classes. Do not write any files: show
both files in your answer.
