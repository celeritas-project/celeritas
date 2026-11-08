# `celeritas::SetupOptions` field guide

Field names follow the installed `<prefix>/include/accel/SetupOptions.hh`,
which is the authority for the user's version (in the source tree:
`src/accel/SetupOptions.hh`). In Celeritas 1.0 this struct will be replaced
by `inp::FrameworkInput`. The conversion function `celeritas::to_inp` shows
how each field maps onto the new input.

Capacities are **per stream** (one stream per Geant4 worker thread by
default). Optical capacities are per process.

## Required

| Field | Guidance |
|-------|----------|
| `opts.make_along_step` | Along-step factory (field propagation). Options are "incomplete" without it. See `fields.md`. |

## I/O

| Field | Guidance |
|-------|----------|
| `opts.output_file` | JSON diagnostics, default `celeritas.out.json`. Empty disables output. Keep it on while integrating: the `troubleshoot` skill reads it. |
| `opts.geometry_file` | Override the geometry with a GDML file. By default Celeritas converts the in-memory Geant4 geometry. Rarely needed. |
| `opts.geometry_output_file` | Write the Geant4 geometry, including SD markers, as GDML. Useful for reproducing framework geometry outside the framework. |
| `opts.physics_output_file` | ROOT dump of imported physics data (needs ROOT). |
| `opts.offload_output_file` | Write offloaded tracks as events (HepMC3, ROOT, or JSON by extension). These can be rerun in `celer-sim` to debug or benchmark without the framework. |

## Stepping capacities and limits

| Field | Guidance |
|-------|----------|
| `opts.max_num_tracks` | Track slots per stream. If unset, Celeritas uses 4096 (CPU) or 1048576 (GPU) slots in total, divided among streams. CPU: about 1-4k per stream. GPU: more slots improve occupancy until device memory runs out. |
| `opts.initializer_capacity` | Initial buffer for queued primaries+secondaries (grows as needed). Default: 8x track slots. The examples use 32-128x. |
| `opts.secondary_stack_factor` | Secondary storage per track slot (default 2). When it fills, interactions are deferred to the physics "failure" action and retried, which wastes steps. Raise it for showers with many secondaries per step. |
| `opts.auto_flush` | Number of buffered tracks before transporting mid-event (defaults to `max_num_tracks`). Lower values reduce memory; higher values improve GPU efficiency. |
| `opts.max_steps` | Kill a track after this many steps (default: unlimited). Use it to stop looping tracks. |
| `opts.max_step_iters` | Abort the stepping loop after this many iterations (default: unlimited). Combine with `CELER_NONFATAL_FLUSH` in production. |
| `opts.track_order` | Track sorting strategy for GPU performance. Leave the default unless profiling says otherwise. |
| `opts.get_num_streams` | Override the number of streams (default: the run manager's thread count). Needed for custom or task-based run managers. |
| `opts.max_field_substeps` | Limit on field-propagation substeps per step (default 10). |

## Physics

| Field | Guidance |
|-------|----------|
| `opts.ignore_processes` | Geant4 process names not imported into Celeritas. The examples ignore `CoulombScat`. |
| `opts.offload_particles` | Unset (default): offload e-, e+, and gamma. An explicit list restricts the offloaded particles. An empty list offloads nothing, used only for optical-distribution offload. |
| `opts.interpolation` | Physics grid interpolation options. Leave the default. |

## Sensitive detectors

`opts.sd`: see `sensitive-detectors.md`.

## Optical photons

`opts.optical` (optional) sets the optical photon capacity, generator,
tracking limits, and detector callbacks. This interface is still evolving:
read the installed `SetupOptions.hh` and `inp/OpticalPhysics.hh`, and the
optical documentation at
https://celeritas-project.github.io/celeritas/user/implementation/optical-physics.html

## Device

| Field | Guidance |
|-------|----------|
| `opts.cuda_stack_size` | Per-thread GPU stack in bytes. VecGeom on GPU often needs more than the default; increase it after a stack overflow or "unspecified launch failure". |
| `opts.cuda_heap_size` | Dynamic GPU heap in bytes (VecGeom). |
| `opts.action_times` | Synchronize after each kernel to time actions. This slows the run; use it only for profiling. |

## Diagnostics (experimental)

| Field | Guidance |
|-------|----------|
| `opts.slot_diagnostic_prefix` | Write per-slot occupancy diagnostics (plot with `scripts/user/plot-slot-diagnostic.py` from the Celeritas source). |
| `opts.add_user_actions` | Callback receiving `CoreParams` to insert custom device actions (see `example/offload-template/src/StepDiagnostic.hh` and the `library-api` skill's `references/device-actions.md`). The API may change between releases. |
