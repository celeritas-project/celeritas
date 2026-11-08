# Validating an integration

Celeritas reproduces Geant4 EM physics statistically, not step for step.
Validate with distributions over many events, never by comparing single
events or random-number streams.

## Switches

| Variable | Effect | Use |
|----------|--------|-----|
| `CELER_DISABLE=1` | Celeritas is not set up; Geant4 transports everything | Reference run with the *same* binary |
| `CELER_KILL_OFFLOAD=1` | Offloadable tracks are killed immediately | Upper bound on achievable speedup (Geant4 time without EM) |
| `CELER_DISABLE_DEVICE=1` | Run Celeritas on CPU in a GPU build | Separate CPU/GPU issues from physics issues |
| `CELER_LOG=debug` | Verbose setup and run logging | First runs of a new integration |

`CELER_DISABLE` and `CELER_KILL_OFFLOAD` are mutually exclusive: setting both
is a setup error.

## Procedure

1. Pick observables the experiment cares about, e.g. total and
   per-subdetector energy deposition, hit multiplicity, shower profiles,
   and timing distributions for time-sensitive detectors.
2. Run N events (enough for percent-level statistics) with `CELER_DISABLE=1`,
   then the same macro with Celeritas. Use the same primary generator
   settings, including different seeds per run if the generator allows it.
3. Compare means with their statistical uncertainties, and compare shapes
   (e.g. a Kolmogorov-Smirnov or chi-square test on histograms). Look at
   ratios per subdetector, not only totals.
4. Explain any significant difference before declaring success. Common
   causes:
   - The field in `opts.make_along_step` differs from the Geant4 field.
   - `opts.ignore_processes` or the physics list differ from what was
     intended. Check the physics summary in `CELER_LOG=debug` output.
   - SD options drop data the SD uses (see `sensitive-detectors.md`).
   - Truth or hit code that depends on Geant4 tracking/stepping actions,
     which Celeritas-created tracks never trigger.
   - Production cuts or user limits (`G4UserLimits`) applied differently in
     the two runs.
5. Record timing from the Celeritas output file (`result.time` in
   `celeritas.out.json`; `troubleshoot` has a summarizer) and from the
   application, so the speedup claim uses the same event sample.

## Reproducibility

Celeritas results are reproducible for a fixed configuration and seed, but
they can change with track-slot counts, thread counts, and the
`CELERITAS_RESEED` build option. Do not expect bitwise agreement between
CPU and GPU runs, or with Geant4.
