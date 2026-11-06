---
type: llm
---

The reviewed application targets Celeritas 0.7 (`find_package(Celeritas 0.7
REQUIRED)` in its `CMakeLists.txt`) and uses `TrackingManagerIntegration`. In
0.7 this integration sets up and tears down automatically, so it needs no user
run action, and it has no `BuildForMaster` or `Build` methods.

FAIL if the response reports the absence of a run action,
`BeginOfRunAction`/`EndOfRunAction` calls, or `BuildForMaster` calls as a
problem to fix, or recommends adding them.

PASS otherwise. Mentioning that they are not needed is fine.
