# `/celer/` Geant4 UI commands

The integration classes create a `SetupOptionsMessenger` automatically, so
these commands work in any Geant4 macro. They change the options passed to
`SetOptions`, and they must run **before `/run/initialize`**. The list comes
from `src/accel/SetupOptionsMessenger.cc`. Run `/control/manual /celer/` in
the user's application to list the commands their version supports.

| Command | Option |
|---------|--------|
| `/celer/geometryFile <gdml>` | `opts.geometry_file` |
| `/celer/outputFile <json>` | `opts.output_file` |
| `/celer/physicsOutputFile <root>` | `opts.physics_output_file` |
| `/celer/offloadOutputFile <file>` | `opts.offload_output_file` |
| `/celer/geometryOutputFile <gdml>` | `opts.geometry_output_file` |
| `/celer/maxNumTracks <n>` | `opts.max_num_tracks` |
| `/celer/maxNumSteps <n>` | `opts.max_steps` |
| `/celer/maxInitializers <n>` | `opts.initializer_capacity` |
| `/celer/secondaryStackFactor <x>` | `opts.secondary_stack_factor` |
| `/celer/autoFlush <n>` | `opts.auto_flush` |
| `/celer/maxFieldSubsteps <n>` | `opts.max_field_substeps` |
| `/celer/slotDiagnosticPrefix <prefix>` | `opts.slot_diagnostic_prefix` |
| `/celer/detector/enabled <bool>` | `opts.sd.enabled` |
| `/celer/device/stackSize <bytes>` | `opts.cuda_stack_size` |
| `/celer/device/heapSize <bytes>` | `opts.cuda_heap_size` |
| `/celer/device/actionTimes <bool>` | `opts.action_times` |

`/celer/cuda/...` is a deprecated alias of `/celer/device/...`. Prefer the
new directory.

Options that take C++ objects have no macro: the field factory
(`opts.make_along_step`), process and particle lists, SD volume sets, and
optical options. These must be set in code. If the code sets nothing,
`SetOptions` warns that the input is incomplete, and macros must supply the
rest before `/run/initialize`.

Example macro fragment for a GPU run:

```
/celer/maxNumTracks 262144
/celer/maxInitializers 2097152
/celer/outputFile celeritas-gpu.out.json
/run/initialize
```
