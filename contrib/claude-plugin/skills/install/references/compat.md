# Compatibility constraints

The authoritative list for a given version is the Celeritas `README.md`
("compatibility" section) and the CI configurations. As of Celeritas 0.7:

## Geant4

| Geant4 | Support |
|--------|---------|
| 10.5-10.7 | Builds and runs; **no tracking-manager offload** (use `UserActionIntegration`) |
| 11.0 | Tracking-manager offload; **no fast-simulation offload** |
| 11.1-11.3 | Full support |
| 11.4 | Supported with known issues (see the README for the tracking issue link) |

Celeritas must be built against the same Geant4 installation (version and
build options such as multithreading) that the application uses.

## Geometry engine

| Engine | CUDA | HIP | CPU | Notes |
|--------|------|-----|-----|-------|
| VecGeom (1.2.10+ or 2.2+, with GDML) | yes | **no** | yes | Default. May need larger CUDA stack/heap. Not with `real_type=float`. |
| ORANGE | yes | yes | yes | Native Celeritas engine, converted from Geant4/GDML |
| Geant4 | **no** | **no** | yes | Reference/debugging; no OpenMP |

## Other

- CUDA and HIP are mutually exclusive.
- OpenMP track parallelism should not be combined with multithreaded Geant4.
- Perfetto profiling is CPU-only.
- Compilers tested in CI include recent GCC, Clang, and NVCC versions, and
  C++17 through C++23. `cxxstd` must match Geant4's.
- GPU architectures must include the target device's compute capability
  (or `gfx` target). Otherwise kernels fail at run time with a warning that
  code "may mysteriously die".
