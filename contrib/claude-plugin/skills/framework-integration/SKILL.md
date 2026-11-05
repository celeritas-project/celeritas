---
name: framework-integration
description: Integrate Celeritas into an experiment simulation framework rather than a plain Geant4 main() - DD4hep/DDG4 and ddsim (Key4hep, EIC, FCC), LArSoft/art for DUNE optical photons, or custom/task-based frameworks such as Gaudi, CMSSW, or Athena. Covers where to register the tracking manager and set options, configuration plumbing, thread/stream mapping, logging and exception routing, GPU assignment per process, MC-truth limits, and reproducing framework jobs outside the framework.
---

Frameworks own `main()`, the run manager, the physics list, and threading, so
the work is finding the right hooks. The Celeritas calls are the same as in
`geant4-offload`. Read that skill's references for option and SD details.

## 1. Check prerequisites

1. Run or recall `doctor`. The framework's Geant4 must be the one Celeritas
   was built with. DD4hep and LArSoft support need `CELERITAS_USE_DD4hep` and
   `CELERITAS_USE_LArSoft` builds.
2. Identify the framework and the integration path:

   | Framework | Path | Read |
   |-----------|------|------|
   | DD4hep/DDG4 (`ddsim`, Key4hep, EIC, FCC, ...) | Bundled `CelerPhysics` DDG4 physics plugin: configure from the steering file, no C++ | [references/dd4hep.md](references/dd4hep.md) |
   | LArSoft/art (DUNE) optical photons | Bundled `PDFullSimCeler` producer module: configure from FHiCL | [references/larsoft.md](references/larsoft.md) |
   | Gaudi, CMSSW, Athena, custom Geant4 drivers | Write a small adapter in the framework's simulation service | [references/task-based.md](references/task-based.md) |

   Before writing new code for DD4hep or LArSoft, check whether the bundled
   plugin covers the need.

## 2. Find the hooks (custom frameworks)

Locate these in the framework source, and quote file and function back to the
user before editing:

1. **Physics list assembly:** where `G4VModularPhysicsList` is built or
   physics constructors are registered. Register
   `celeritas::TrackingManagerConstructor` there, once.
2. **Master-thread setup before `/run/initialize`:** after the run manager
   exists. Call `TrackingManagerIntegration::Instance().SetOptions(...)`
   there, with options built from the framework's configuration system.
3. **Worker threads and streams:** whether workers are Geant4 worker threads
   (with valid `G4Threading::G4GetThreadId()`) and how many exist. Task-based
   run managers may report zero threads: set `opts.get_num_streams`.
4. **Sensitive detectors:** whether SDs are attached on the master thread.
   Celeritas maps SDs from the master. Use `opts.sd.force_volumes`
   otherwise.
5. **Logging and errors:** the framework's message service and its
   `G4VExceptionHandler`.
6. **Truth handling:** any user tracking/stepping actions or trajectory
   storage that build MC truth. These do not run for Celeritas-created
   tracks.

## 3. Implement

Follow [references/task-based.md](references/task-based.md). In short:

- Prefer the high-level `TrackingManagerIntegration`. Use the low-level
  `SharedParams` + `LocalTransporter` pair only if the framework needs to
  own the objects (e.g. no singletons) or drive offload itself.
- Expose the few options users must tune as framework parameters:
  track slots, initializer capacity, ignored processes, field map, output
  file. `src/ddceler/CelerPhysics.cc` is a compact model of this mapping.
- Keep `CELER_DISABLE=1` working. It is the reference run for validation and
  an escape hatch for production.

## 4. Validate

- Run the framework's standard job with and without `CELER_DISABLE=1` and
  compare per-subdetector energy and hit counts (`geant4-offload` skill,
  `references/validation.md`).
- To debug or benchmark outside the framework, set
  `opts.geometry_output_file` (GDML) and `opts.offload_output_file`
  (`.jsonl`, `.root`, or a HepMC3 extension). Then replay the offloaded
  tracks in `celer-sim` with a `"read"` event generator (`standalone-input`
  skill).

## Checklist

- [ ] Bundled plugin used where one exists (DD4hep, LArSoft)
- [ ] Tracking manager registered once on the physics list Geant4 uses
- [ ] `SetOptions` on master, after run manager construction, before initialization
- [ ] Stream count matches the framework's worker threads/tasks
- [ ] SDs visible on master (or `force_volumes` set)
- [ ] Celeritas logs routed to the framework's logger; Geant4 exception handler unchanged
- [ ] One GPU per process mapping understood (`CUDA_VISIBLE_DEVICES` or MPI rank)
- [ ] MC-truth consumers reviewed for Celeritas-created tracks
- [ ] Framework job passes with and without `CELER_DISABLE=1`
