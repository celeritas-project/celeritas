# Celeritas error catalog

Match on the quoted fragment, since the full message includes values that
vary. Celeritas errors raised inside Geant4 appear as fatal `G4Exception`s:

- The exception code names the phase that failed, e.g. `celer.setup`,
  `celer.init.verify`, `celer.init.worker`, `celer.event.begin`.
- "issued by" gives the source location, e.g. `celeritas/ext/GeantSd.cc:<line>`.
- The text starts with `Celeritas runtime error:`, followed by the message.

The first Celeritas error in the log is the one to fix: later errors are often
consequences. If a message is not listed here, search the installed headers or
the source at the user's version for the fragment.

## Any phase

| Message fragment | Cause | Fix |
|------------------|-------|-----|
| "required dependency is disabled in this build" | The requested feature needs an optional package Celeritas was built without (e.g. ROOT output, HepMC3 input) | Use a different output/input format, or rebuild Celeritas with that package (`celeritas.use` from `doctor` lists what is enabled) |
| "feature is not yet implemented" | The configuration asks for something Celeritas does not support yet | Change the configuration; check the release notes or open an issue |

## Geant4 integration setup

| Message fragment | Cause | Fix |
|------------------|-------|-----|
| "no G4 sensitive detectors are defined" | SD callbacks are enabled but the geometry has no SDs on the master thread | Set `opts.sd.enabled = false`, or if SDs exist only on workers, list them in `opts.sd.force_volumes` |
| "tracking manager(s) are not attached correctly" | `TrackingManagerConstructor` missing from the physics list Geant4 actually uses | Register it once on the final modular physics list |
| "not all particles from TrackingManagerConstructor are active in Celeritas" | A particle handed to Celeritas has no Celeritas physics | Check `opts.offload_particles` and the physics list (EM constructor present?) |
| "fast simulation process(es) are not attached correctly" | `G4FastSimulationPhysics` missing or not activated for e-/e+/gamma | Add it and call `ActivateFastSimulation` for each offloaded particle |
| "is too old to support the tracking manager offload interface" | Geant4 < 11.0 | Use `UserActionIntegration`, or upgrade Geant4 |
| "is too old to support the fast simulation offload interface" | Geant4 < 11.1 | Use `TrackingManagerIntegration` |
| "options cannot be set before G4RunManager is constructed" | `SetOptions` called before the run manager exists | Move it after run manager construction |
| "options cannot be set after Celeritas is constructed" | `SetOptions` called after the run started | Call it before `/run/initialize` |
| "SetOptions called with incomplete input" (warning) and "SetOptions or UI entries were not completely set before BeginRun" | `opts.make_along_step` (or another required option) is unset | Set it in code, or via `/celer/` macros before `/run/initialize` |
| "TrackingManagerIntegration auto hooks are active: remove manual Celeritas BeginOfRunAction and EndOfRunAction calls" (warning) | Legacy run-action calls with the tracking manager | Delete those calls |
| "BeginOfRunAction was not called on master thread" | Fast-sim/user-action integration without a master run action | Create the run action in `BuildForMaster` too |
| "cannot call Integration::GetState from master thread" | Accessing per-thread state from master | Call it from worker-thread code only |
| "cannot access shared params when Celeritas is disabled or outside of a run" | `GetParams` called with `CELER_DISABLE` set or outside a run | Guard with `GetMode()` |
| "both CELER_DISABLE and CELER_KILL_OFFLOAD" | Both variables set | Unset one |
| "is not available in Celeritas", "cannot offload '", "has no physics processes defined" (warning) | Unsupported particle in `opts.offload_particles` | Remove it; default offload is e-, e+, gamma |

## Physics and data

| Message fragment | Cause | Fix |
|------------------|-------|-----|
| "environment variable G4LEDATA is not defined" | Geant4 EM data not configured | Source `geant4.sh`/load the Geant4 data module, or export `G4LEDATA` (`geant4-config --datasets` lists the path) |
| "cannot build unsupported EM process" | The physics list has an EM process Celeritas lacks | Add the Geant4 process name to `opts.ignore_processes` (results for that process then come only from Geant4 for non-offloaded particles) |
| "Cannot load unknown process" (error log) or "Encountered unknown process" (warning) | Process not recognized during import | Same as above; check whether it matters for the observables |
| "no Geant4 global field has been set: cannot build magnetic field map" | Map helper ran before the Geant4 field existed | Build the map inside the factory's lazy function; attach the field to the global `G4FieldManager` |

## Runtime

| Message fragment | Cause | Fix |
|------------------|-------|-----|
| "number of step iterations exceeded the allowed maximum" | `opts.max_step_iters` reached: tracks are stuck or looping | Set `opts.max_steps` to kill looping tracks. Export `CELER_NONFATAL_FLUSH=1` to print and kill active tracks instead of aborting. Save `opts.offload_output_file` to reproduce in `celer-sim`. Report geometry/navigation issues upstream |
| "offloaded tracks were not flushed" | Run ended with buffered tracks (missing end-of-event call, or an earlier exception) | With user-action or fast-sim integration, call `EndOfEventAction`; look earlier in the log for the first error |
| "primary buffer capacity of" ... "exceeded" | More primaries queued than the primary capacity (standalone or custom drivers) | Increase the primary capacity (`auto_flush`/`primaries`) |
| "has CUDA compute capability of" ... "code may mysteriously die at runtime" (warning) | GPU architecture not in the compiled `CMAKE_CUDA_ARCHITECTURES` | Rebuild Celeritas (and VecGeom) for this GPU |
| "MPI was not initialized (needed to construct a communicator)" | Celeritas built with MPI, app not MPI-aware | `export CELER_DISABLE_PARALLEL=1`, or initialize MPI in the app |
| CUDA "an illegal memory access was encountered" or "unspecified launch failure" with VecGeom | Often GPU stack overflow in VecGeom navigation | Raise `opts.cuda_stack_size` (or `CUDA_STACK_SIZE`), e.g. doubling until it passes; rerun with `CELER_LOG=debug` |
| Run hangs or is very slow on GPU | Debug build, too few track slots, or synchronizing timers | Check `celeritas.debug` from `doctor`; raise `opts.max_num_tracks`; disable `opts.action_times` |

## CMake and linking

| Message fragment | Cause | Fix |
|------------------|-------|-----|
| "may not be compatible with requested version" ... "please update your find_package version range" | Requested 0.x minor older than the installed one | Raise the minimum or use a version range (see `cmake-link`) |
| "CUDA compiler/linker are required when compiling with CUDA" | Downstream project did not enable CUDA | `enable_language(CUDA)` in the top-level `CMakeLists.txt` |
| "CUDA mismatch between the VecGeom installation" | VecGeom found at configure time differs from the one Celeritas used | Point `CMAKE_PREFIX_PATH` at the matching VecGeom |
| "Could not find a configuration file for package \"Geant4\" that is compatible with requested version" | Different Geant4 in the environment than at Celeritas build time | Load the Geant4 that Celeritas was built with |
| Linker: undefined `__cudaRegisterLinkedBinary` or `nvlink` errors | RDC device link missing | Use `celeritas_target_link_libraries` for every target; for forced `SHARED` plugins link `$<TARGET_NAME_IF_EXISTS:Celeritas::accel_final>` |
