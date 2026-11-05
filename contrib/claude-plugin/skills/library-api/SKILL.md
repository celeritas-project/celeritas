---
name: library-api
description: Use Celeritas as a C++ library outside the Geant4 offload layer - driving standalone transport from code (inp::StandaloneInput, setup::standalone_input, Stepper), loading ORANGE/VecGeom/Geant4 geometry or particle/material/physics data directly, collecting step data without Geant4 sensitive detectors, writing custom GPU step actions, and the params/states/collections data model. Use when a user writes C++ that includes Celeritas headers other than the accel Geant4 integration.
---

Celeritas' most stable public surfaces are the Geant4 integration (`accel/`)
and the `inp::` input structs. Lower-level classes (the stepping loop,
Collections, actions) are public but **can change between 0.x releases**.
For example, `Stepper` recently gained an asynchronous interface and its
synchronous call operator is marked deprecated. So for each pattern below,
read the *installed* headers and copy from the reference code **at the
user's version** rather than from memory.

## 1. Check that the library is really needed

| Goal | Better path |
|------|-------------|
| Run a problem defined in JSON | `celer-sim` / `celer-optical` (`standalone-input` skill) |
| Accelerate a Geant4 application or framework | `geant4-offload` / `framework-integration` |
| Embed transport in a non-Geant4 program, use geometry or physics data from C++, add GPU-side scoring | This skill |

Run or recall `doctor`: the build's geometry engine, GPU support, units, and
`real_type` determine which classes exist and how they behave.

## 2. Build setup

- Link with `celeritas_target_link_libraries` (see `cmake-link`):
  `Celeritas::celeritas` for physics and transport, or a subset for
  lower-level use (`Celeritas::orange`, `Celeritas::geocel`,
  `Celeritas::corecel`).
- In `main`, initialize MPI and the GPU before building any params:
  `celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);` (header
  `corecel/sys/ScopedMpiInit.hh`), then `celeritas::activate_device();`
  (header `corecel/sys/Device.hh`). `example/data-model/data-model.cc` is the
  minimal complete program.
- Include what you use: many headers only forward-declare their
  collaborators. Check the include paths against the installed tree
  (`<prefix>/include/`) instead of guessing.
- Device code (`CELER_FUNCTION` executors, kernels) goes in a `.cc`/`.cu`
  pair. Compile the `.cu` only when `CELERITAS_USE_CUDA` or
  `CELERITAS_USE_HIP` is set.

## 3. Pick the pattern

| Pattern | Entry points (installed headers) | Copy from |
|---------|----------------------------------|-----------|
| Data model: params (shared, immutable), states (per track slot), `Collection`, views | `corecel/data/Collection.hh`, e.g. `celeritas/phys/ParticleParams.hh` | `example/data-model/data-model.cc`; manual: `doc/implementation/data-model.rst` |
| Programmatic standalone transport | `celeritas/inp/StandaloneInput.hh`, `celeritas/setup/StandaloneInput.hh`, `celeritas/global/Stepper.hh` | `app/celer-sim/Runner.cc`, `app/celer-sim/Transporter.cc`; see [references/standalone-transport.md](references/standalone-transport.md) |
| Geometry only (navigation, volume lookup, ray tracing) | `orange/OrangeParams.hh` (`from_gdml`, `from_geant`, `from_json`), `geocel/vg/VecgeomParams.hh`, `geocel/GeantGeoParams.hh` | `app/celer-geo/Runner.cc` |
| Step data without Geant4 SDs (calorimetry, MC truth) | `celeritas/user/StepCollector.hh`, `celeritas/user/StepInterface.hh`; built-ins `SimpleCalo`, `ActionDiagnostic`, `StepDiagnostic`, `RootStepWriter` | `src/celeritas/user/` |
| Custom GPU step action | `celeritas/global/ActionInterface.hh`, `celeritas/global/TrackExecutor.hh`, `celeritas/global/ActionLauncher.hh` | `example/offload-template/src/StepDiagnostic*`; see [references/device-actions.md](references/device-actions.md) |

Notes:
- `OrangeParams::from_gdml` uses Geant4 to read GDML. Without Geant4 it looks
  for a matching `.org.json` file instead. The volume count is
  `geo->volumes()->num_volumes()`, which needs `geocel/VolumeParams.hh`
  because `orange/OrangeParams.hh` only forward-declares `VolumeParams`.
- Geant4 geometry and physics import require a Celeritas build with Geant4.
  Only one Geant4 world can be loaded per process.
- Units: Celeritas uses its native unit system (`celeritas.units`).
  `celeritas/Units.hh` and the `Quantity` helpers (`value_as`,
  `native_value_from`) convert explicitly.

## 4. Checklist

- [ ] Library use justified (not achievable with `celer-sim` JSON or `accel`)
- [ ] `ScopedMpiInit` + `activate_device()` before any params
- [ ] Linked with `celeritas_target_link_libraries`; `.cu` compiled only for GPU builds
- [ ] API calls checked against the installed headers of the user's version
- [ ] Reference code copied from the matching release tag, not `develop`
- [ ] Device code follows the rules in `references/device-actions.md`
