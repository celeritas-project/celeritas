---
name: doctor
description: Detect the installed Celeritas version and build configuration (CUDA/HIP, VecGeom/ORANGE/Geant4 geometry, MPI, debug), the Geant4 version and EM data, and GPU visibility. Use before giving Celeritas install, CMake, Geant4 integration, or runtime advice, and when the user asks whether Celeritas is set up correctly.
allowed-tools:
  - Bash(python3 -I ${CLAUDE_SKILL_DIR}/celer-doctor.py)
  - Bash(python3 -I ${CLAUDE_SKILL_DIR}/celer-doctor.py *)
---

Ground every recommendation in what is actually installed. Celeritas options
and APIs change between versions (for example, `SetupOptions` is being
replaced by `inp::FrameworkInput` on the way to 1.0), and GPU, geometry, and
MPI choices are fixed when Celeritas is built.

## 1. Probe

```bash
python3 -I ${CLAUDE_SKILL_DIR}/celer-doctor.py
```

Pass the install prefix as an argument if the user names one or their build
uses one (e.g. from `Celeritas_DIR` in their `CMakeCache.txt`). The probe is
read-only and always exits 0. It prints `key=value` facts, then `warn=` and
`note=` lines, then the raw `celer-config` JSON.

If nothing is found, ask how Celeritas is provided (Spack environment, module,
container, custom prefix) before guessing; the `install` instructions in the
user documentation cover new installs:
https://celeritas-project.github.io/celeritas/user/usage/installation.html

## 2. Interpret

| Fact | What it decides |
|------|-----------------|
| `celeritas.version` | Which docs and API apply. Read the *installed* headers under `<prefix>/include/` (e.g. `accel/SetupOptions.hh`, `accel/TrackingManagerIntegration.hh`, `celeritas/inp/*.hh`) as the API ground truth instead of relying on memory. |
| `celeritas.use` | `geant4` is required for `Celeritas::G4` and offload. `cuda`/`hip` means GPU builds (and downstream CMake must enable that language). `mpi` may need `CELER_DISABLE_PARALLEL=1` in non-MPI apps. `hepmc3`/`root` enable event and output I/O. |
| `celeritas.core_geo` | `VecGeom` (GPU-capable, may need a larger CUDA stack), `ORANGE` (native, works with HIP), or `Geant4` (CPU only). |
| `celeritas.gpu_architectures` vs `device.capability` | A mismatch makes kernels fail at run time; Celeritas must be rebuilt for the GPU. Downstream CMake inherits these architectures by default. |
| `celeritas.debug` | Debug builds are far slower: never benchmark with them. |
| `geant4.version` vs `celeritas.geant4_built_with` | Must match. Geant4 >= 11.0 enables tracking-manager offload; >= 11.1 enables fast-simulation offload. |
| `env.*` | Pre-set variables that silently change behavior, e.g. `CELER_DISABLE`, `CELER_KILL_OFFLOAD`, `CELER_DISABLE_DEVICE`. |

Treat every `warn=` line as a problem to raise with the user; `note=` lines
are context.

## 3. Report

Summarize in at most ten lines: version and prefix, GPU and geometry
configuration, Geant4 version, each problem found with its fix, and the
recommended next step. Remember these facts for the rest of the session
instead of re-running the probe.
