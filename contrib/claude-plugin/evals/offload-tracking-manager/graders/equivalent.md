---
type: llm
focus: { source: file, path: trackingmanager-offload.cc }
---

The file is a Geant4 application that should now offload EM tracks to
Celeritas through its tracking manager integration (Celeritas 0.7).

PASS if all of the following hold:
- It includes `<CeleritasG4.hh>` (or the equivalent accel headers).
- It obtains `celeritas::TrackingManagerIntegration::Instance()` and
  registers `new celeritas::TrackingManagerConstructor(&tmi)` (or an
  equivalent pointer to the integration) on the physics list passed to the
  run manager.
- It calls `SetOptions(...)` on the integration once, after the run manager
  is constructed and before `Initialize()`.
- It does not add `BeginOfRunAction`/`EndOfRunAction` calls for the tracking
  manager integration, and does not remove or break the existing
  application logic (detector, sensitive detector, event checks).

FAIL if any of these is missing or wrong.
