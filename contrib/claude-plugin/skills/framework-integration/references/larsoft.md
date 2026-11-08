# LArSoft / art (DUNE optical photons)

Celeritas provides art modules (built with `CELERITAS_USE_LArSoft`, sources
in `src/larceler/`). The main one is the `PDFullSimCeler` producer: a
replacement for LArSim's `PDFastSimPAR` that runs full optical-photon
transport on CPU or GPU.

How it differs from Geant4 offload:

- It runs **after** LArG4 as a separate producer module. It is independent
  of the Geant4 run manager, so there is no tracking manager and no
  `SetupOptions`.
- Input: `sim::SimEnergyDeposit` steps from the `IonAndScint` producer
  (`SimulationLabel`).
- Output: photon-detector responses with backtracking metadata, i.e. what
  `PDFastSimPAR` would produce.
- The GDML must contain optical properties (bulk and surface) and mark the
  photon detectors as sensitive. `example/larceler/setup-dune-gdml.sh` and
  its patches show how the DUNE 10 kt GDML is prepared.

## Environment

After installing Celeritas against the matching LArSoft (FNAL Spack or UPS,
see the installation manual), load its paths:

```sh
eval $(<prefix>/bin/larceler-env)      # sh/bash/zsh
eval `<prefix>/bin/larceler-env -c`    # csh/tcsh
```

This prepends `CMAKE_PREFIX_PATH`, the library path, `CET_PLUGIN_PATH`, and
`FHICL_FILE_PATH`, and sets `CELERITAS_DIR`. With Spack-distributed LArSoft,
jobs may also need the current directory in `FHICL_FILE_PATH`, and
`FW_SEARCH_PATH` updated for a nonstandard GDML.

## FHiCL

```
#include "pdfullsimceler.fcl"

physics: {
  producers: {
    celersim: @local::PDFullSimCeler      # or @local::PDFullSimCelerGPU
  }
  end_path: [ celersim ]
}
```

The parameters and their defaults are in the installed
`pdfullsimceler.fcl` (source: `src/larceler/pdfullsimceler.fcl`). Read it
rather than relying on memory. The main groups:

| Parameters | Purpose |
|------------|---------|
| `SimulationLabel` | Producer label of the energy deposits (default `IonAndScint`) |
| `EnableDevice`, `DeviceStackSize`, `DeviceHeapSize` | GPU on/off and device limits (the `PDFullSimCelerGPU` table enables the GPU and enlarges capacities) |
| `OpticalCapacityPrimaries`, `OpticalCapacityTracks`, `OpticalCapacityGenerators` | Optical buffer sizes. GPU runs need far larger values than CPU runs. |
| `OpticalLimitSteps`, `OpticalLimitStepIters` | Kill/abort limits (0 = unlimited) |
| `OutputFile` | Celeritas diagnostics, JSON lines (`celeritas.out.jsonl`). The `troubleshoot` summarizer reads this format. |
| `ActionTimes`, `StepTimes` | Timing diagnostics |
| `Seed` | Random number seed |

## Workflow and analysis

`example/larceler/` contains a complete chain (GENIE, LArG4, fast vs
Celeritas optical simulation, analysis) driven by `run.sh`, with
`dune10kt_1x2x6_cpu.fcl` and `opticalsim_celeritas_dune10kt_1x2x6.fcl` as
job examples. The `PDSimAna` module (`pdsimana.fcl`, `pdsimana_job.fcl`)
compares photon-detector results. `GeoSimExporter` exports geometry and
energy-deposition data for testing.

## GPU notes

- Set `CUDAARCHS` (e.g. `export CUDAARCHS=80`) when *building* Celeritas
  with the LArSoft preset. The preset enables CUDA but cannot know the card.
- Old SL7 Apptainer images may fail to load LArG4 libraries with `--nv` due
  to host OpenGL libraries. The installation manual describes the
  `nvliblist.conf` workaround.
