---
name: troubleshoot
description: Diagnose Celeritas failures and surprises - G4Exception/Celeritas runtime errors, CMake or link errors, GPU crashes, stuck or slow runs, missing hits - and interpret celeritas.out.json diagnostic output. Use when a user pastes a Celeritas error or log, a Celeritas-enabled build or run fails, or results differ from pure Geant4.
allowed-tools:
  - Bash(python3 -I ${CLAUDE_SKILL_DIR}/summarize-output.py *)
---

Work from evidence: the **first** Celeritas error in the log, the build
configuration, and the diagnostic JSON. Do not guess from the symptom alone.

## 1. Collect

- The installed configuration from `doctor` (run it if not yet done).
- The first error. Celeritas errors in Geant4 apps are fatal `G4Exception`s
  with a `celer.*` code and the text `Celeritas runtime error: ...`. Ask for
  the log from the start of the run if only the end was shared.
- The diagnostic output: `celeritas.out.json` by default (`opts.output_file`
  or `/celer/outputFile`). Integrations write it at the end of a run, and the
  Celeritas apps (`celer-sim`, `celer-g4`, ...) also write it when they fail
  with an exception. A fatal `G4Exception` in a user application may abort
  before the file is written; then work from the log alone.

  ```bash
  python3 -I ${CLAUDE_SKILL_DIR}/summarize-output.py celeritas.out.json
  ```

  It reports the build and device, the environment variables Celeritas read
  (and their values), any recorded exception with its context (e.g. the
  state of the failing track), timing, and kernel statistics.

## 2. Match

Look the message up in [references/error-catalog.md](references/error-catalog.md).
Its sections cover setup, physics data, runtime, and CMake/linking. For
messages not in the catalog, search the user's *installed* headers or the
Celeritas source at their version for the message text. The surrounding
`CELER_VALIDATE` condition usually explains the cause.

## 3. Narrow down

Use switches that leave the code unchanged:

| Question | Experiment |
|----------|------------|
| Is it Celeritas at all? | `CELER_DISABLE=1` (pure Geant4, same binary) |
| GPU-specific? | `CELER_DISABLE_DEVICE=1` (Celeritas on CPU) |
| What happens during setup? | `CELER_LOG=debug` and `CELER_LOG_LOCAL=debug` |
| Messages from all threads? | `CELER_LOG_ALL_LOCAL=1` |
| Need the run to finish despite stuck tracks? | `CELER_NONFATAL_FLUSH=1` (kills and prints active tracks instead of aborting) |
| Reproduce outside a framework? | Set `opts.offload_output_file` and `opts.geometry_output_file`, then rerun the saved tracks and GDML in `celer-sim` |
| MPI startup problems? | `CELER_DISABLE_PARALLEL=1` |

The full list of variables is in
https://celeritas-project.github.io/celeritas/user/usage/execution/environment.html
(`doc/usage/execution/environment.rst`).

For results that differ from Geant4, follow the validation steps in the
`geant4-offload` skill (`references/validation.md` there). Check the field,
ignored processes, SD options, and MC-truth assumptions before suspecting
physics.

## 4. Report

State the cause, the evidence (quoted message or JSON entry), and the fix as
a concrete edit or command. If it looks like a Celeritas bug (an assertion
`DebugError`, a crash in Celeritas code with a valid configuration, or
navigation failures), help the user file an issue at
https://github.com/celeritas-project/celeritas/issues with:

- `celer-config --device` output (or the `doctor` summary),
- the first error and its surrounding log,
- the summarized `celeritas.out.json`,
- a reproducer, ideally the offloaded tracks plus GDML from
  `opts.offload_output_file`/`opts.geometry_output_file`.
