---
name: geant4-offload
description: Add Celeritas GPU/CPU offloading of electromagnetic tracks (e-, e+, gamma) to an existing Geant4 application, or adjust an existing integration - TrackingManagerIntegration, FastSimulationIntegration, or UserActionIntegration, SetupOptions, sensitive detector hit callbacks, magnetic fields, and /celer/ UI macros. Use when a user wants to accelerate a Geant4 app with Celeritas or edits code that calls these classes.
---

Celeritas transports the EM particles that Geant4 hands it, then sends hits
back to the application's Geant4 sensitive detectors (SDs). Integration is a
few lines in `main()` plus a `celeritas::SetupOptions` struct. The changes
for a minimal app are in `example/geant4/add-celer.diff`.

## 1. Check prerequisites

1. Run or recall `doctor`. `celeritas.use` must contain `geant4`, and the
   Geant4 in the environment must be the one Celeritas was built with.
2. Choose the integration from the Geant4 version:

   | Geant4 | Integration | Setup/teardown |
   |--------|-------------|----------------|
   | >= 11.0 (recommended) | `TrackingManagerIntegration` + `TrackingManagerConstructor` | automatic (Geant4 state hooks) |
   | >= 11.1, offload only in a region | `FastSimulationIntegration` + `FastSimulationModel` + `G4FastSimulationPhysics` | manual run-action calls |
   | older | `UserActionIntegration` from user tracking/event/run actions | manual run-action calls |

   Avoid the fast-simulation route unless region-limited offload is required:
   tracks that leave the region stay in Celeritas.
3. Link the app with `Celeritas::G4` (see the `cmake-link` skill).

## 2. Edit the application (tracking manager)

In the file that builds the run manager (usually `main`):

```cpp
#include <CeleritasG4.hh>

auto& tmi = celeritas::TrackingManagerIntegration::Instance();
physics_list->RegisterPhysics(new celeritas::TrackingManagerConstructor(&tmi));
run_manager->SetUserInitialization(physics_list);
// ... other user initializations ...
tmi.SetOptions(MakeCelerOptions());  // after the run manager exists
run_manager->Initialize();
```

- Call `SetOptions` once, on the master thread, after the run manager is
  constructed and before `/run/initialize`. Celeritas reports "options
  cannot be set before G4RunManager is constructed" or "options cannot be set
  after Celeritas is constructed" otherwise.
- Register `TrackingManagerConstructor` on the same physics list Geant4
  uses, exactly once. For experiment frameworks (DD4hep, LArSoft, Gaudi,
  CMSSW, Athena, ...) use the `framework-integration` skill instead.
- Run actions depend on the Celeritas version:
  - **Celeritas 0.7 and later:** the tracking manager integration sets up
    and tears down automatically. Remove legacy `BeginOfRunAction`/
    `EndOfRunAction` calls to it: they are no-ops that print a warning.
  - **Celeritas 0.6:** call `tmi.BeginOfRunAction(run)` and
    `tmi.EndOfRunAction(run)` from a user run action, created for both
    master (`BuildForMaster`) and workers.
  - The fast-simulation and user-action integrations always require these
    calls. See `example/geant4/fastsim-offload.cc` and
    `example/geant4/simple-offload.cc`.

## 3. Write the options

Model `MakeCelerOptions()` on
`example/offload-template/src/MakeCelerOptions.cc`. Set at least:

- `opts.make_along_step`: field propagation. It is required, and options
  without it are "incomplete" until UI macros fill them. Use
  `celeritas::UniformAlongStepFactory()` for zero field. Read
  [references/fields.md](references/fields.md) for uniform and map fields.
- `opts.max_num_tracks` and `opts.initializer_capacity`: per-stream
  capacities. Use small values on CPU (thousands of tracks) and large values
  on GPU (hundreds of thousands per stream).
- `opts.ignore_processes`: Geant4 processes Celeritas should not try to
  import (the examples use `{"CoulombScat"}`).
- `opts.sd`: if the app has no SDs, set `opts.sd.enabled = false`. Otherwise
  enable only the step data the SDs read. See
  [references/sensitive-detectors.md](references/sensitive-detectors.md),
  including the MC-truth caveats.

Look up every other field in
[references/setup-options.md](references/setup-options.md) and in the
*installed* `<prefix>/include/accel/SetupOptions.hh`. Most fields can
alternatively be set from macros: see [references/ui-macros.md](references/ui-macros.md).

## 4. Build, run, validate

1. Build, then run a few events with `CELER_LOG=debug` and look for the setup
   summary and any warnings.
2. Run the same macro with `CELER_DISABLE=1` (pure Geant4) and compare
   physics observables. Follow [references/validation.md](references/validation.md).
3. If something fails, use the `troubleshoot` skill. For speed, use
   `performance-tuning`. To have the integration reviewed, delegate to the
   `integration-reviewer` agent.

## Checklist

- [ ] Integration class matches the Geant4 version
- [ ] `TrackingManagerConstructor` registered once; `SetOptions` before `/run/initialize`
- [ ] `opts.make_along_step` matches the Geant4 field (including zero field)
- [ ] Capacities sized for CPU or GPU
- [ ] SD options minimal, or `opts.sd.enabled = false` without SDs
- [ ] App still runs with `CELER_DISABLE=1`
- [ ] Results compared against pure Geant4
