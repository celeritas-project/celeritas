---
description: Fix plain target_link_libraries against a CUDA+VecGeom Celeritas
tags: [smoke, cmake]
runs: 2
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill]
---

Our Celeritas install was built with CUDA and VecGeom. Our project's
`CMakeLists.txt` is:

```cmake
cmake_minimum_required(VERSION 3.18)
project(mysim LANGUAGES CXX CUDA)
find_package(Celeritas 0.7 REQUIRED)
find_package(Geant4 REQUIRED)

add_library(mydet SHARED src/Detector.cc src/CelerSetup.cc)
target_link_libraries(mydet PUBLIC Celeritas::G4 ${Geant4_LIBRARIES})

add_executable(mysim main.cc)
target_link_libraries(mysim PRIVATE mydet)
```

Linking `mysim` fails with errors like
`undefined reference to '__cudaRegisterLinkedBinary_..._accel_...'`.
Show the corrected `CMakeLists.txt` and explain the fix briefly.
