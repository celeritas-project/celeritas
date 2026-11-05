---
name: standalone-input
description: Write, update, or debug input for the standalone Celeritas applications - celer-sim (EM transport from JSON), celer-optical (optical photons), celer-g4 (Geant4 app with offload), celer-geo (geometry ray tracing), and celer-export-geant (export Geant4 physics). Use when a user wants to run Celeritas without their own Geant4 application, benchmark or validate a problem, replay offloaded tracks, or fix a celer-sim JSON file.
---

The standalone apps read JSON built from the `celeritas::inp` structs. The
authority for field names and meanings is the user's *installed*
`celeritas/inp/*.hh` (in the source tree: `src/celeritas/inp/`). The app's own
`default` output shows the current structure.

## 1. Pick the app

| Goal | App | Input |
|------|-----|-------|
| EM shower transport benchmark or validation, replaying offloaded tracks | `celer-sim` | JSON `inp::StandaloneInput` |
| Optical photons only | `celer-optical` | JSON `inp::OpticalStandaloneInput` |
| Full Geant4 app with offload, for Geant4-vs-Celeritas comparisons with SD hits | `celer-g4` | JSON (or legacy macro) |
| Inspect or raytrace geometry, compare geometry engines | `celer-geo` (usually through the `celerpy` Python front end) | JSON lines |
| Save Geant4 physics tables for runs without Geant4 | `celer-export-geant` | GDML + physics options JSON |

Run or recall `doctor` first. GPU, geometry engine, ROOT and HepMC3
availability all depend on the build.

## 2. Start from the defaults

```bash
celer-sim default > input.json          # full default StandaloneInput
celer-optical default > optical.json
celer-g4 --dump-default > g4-input.json
celer-export-geant --dump-default > physics-options.json
celer-sim update old-input.json > input.json   # convert deprecated input
```

`celer-sim` warns "Deprecated celer-sim input format" for old-style input.
Convert it with `update` rather than editing by hand.

## 3. Edit

The minimal `celer-sim` input needs a geometry and primaries. The example in
the repository `README.md` ("Running a standalone Celeritas app") generates
10 MeV electrons from a GDML file. Common edits:

| Task | Where |
|------|-------|
| Geometry | `problem.model.geometry`: path to GDML |
| **Run on the GPU** | `"system": {"device": {}}`. Without it, `celer-sim` runs on CPU even in a CUDA build. Set `stack_size`/`heap_size` inside for VecGeom. |
| Primaries | `events.generator` with `"_type": "primary"` (sampled), `"read"` (`event_file`: `.jsonl`, `.root`, or a HepMC3 file), or `"sample"` (ROOT) |
| Replay tracks offloaded from an application | `"_type": "read"` with the file from `opts.offload_output_file`, and the GDML from `opts.geometry_output_file` as geometry |
| Magnetic field | `problem.field`, e.g. `{"_type": "uniform", "strength": [0, 0, 1]}` (tesla); also `rzmap`, `cylmap`, `cartmap`, `none` |
| Physics source | `physics_import`: `{"_type": "geant", "ignore_processes": [...]}` (needs Geant4) or `{"_type": "file", "input": "physics.root"}` |
| Capacities and streams | `problem.control.capacity` (totals over all streams) and `problem.control.num_streams` |
| Output | `problem.diagnostics.output_file` (`"-"` = stdout, the default) |
| Timing | `problem.diagnostics.timers` (`action`, `step`) |
| Environment overrides | `system.environment` (a map of `CELER_*` variables) |

Read [references/json-conventions.md](references/json-conventions.md) for
how variants, optionals, enums, and units appear in JSON. In particular,
omitting an optional, `null`, and `{}` mean three different things.

## 4. Run and check

```bash
CELER_LOG=debug celer-sim input.json > output.json
```

- With `output_file` `"-"`, stdout carries only the JSON result, so it can
  be piped to `jq` or Python. Logs go to stderr.
- Summarize the result with the `troubleshoot` skill's
  `summarize-output.py` (it understands `result.runner`).
- Misspelled keys may be ignored rather than rejected. If a setting has no
  effect, compare the key path against `celer-sim default`.

## References

- Input manual: https://celeritas-project.github.io/celeritas/user/usage/input.html
- JSON conventions: https://celeritas-project.github.io/celeritas/user/usage/execution/json-input.html
- Apps: https://celeritas-project.github.io/celeritas/user/usage/execution/applications.html
- `app/celer-sim/simple-driver.py` (generate inputs from Python)
