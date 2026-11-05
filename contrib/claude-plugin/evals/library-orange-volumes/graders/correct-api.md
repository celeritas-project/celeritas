---
type: llm
---

The answer should contain a complete `main.cc` and `CMakeLists.txt` for a
small program that uses Celeritas as a library.

PASS if both files are shown, `CMakeLists.txt` calls
`find_package(Celeritas ...)`, and the program does not use the Celeritas
Geant4 offload layer (no `accel/` headers, `TrackingManagerIntegration`,
`SetupOptions`, or `CeleritasG4.hh`).

FAIL if either file is missing, `find_package(Celeritas ...)` is absent, or
the offload layer is used.
