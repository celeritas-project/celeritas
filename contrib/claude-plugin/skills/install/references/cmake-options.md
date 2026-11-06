# Building from source with CMake

Use this route when Geant4, ROOT, and so on come from an experiment stack
(CVMFS/LCG, Key4hep, a container) and must be reused. The full option list is
in the installation manual (`doc/usage/installation.rst`). `ccmake` shows
every option with documentation.

```bash
git clone --branch vX.Y.Z https://github.com/celeritas-project/celeritas.git
cmake -S celeritas -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install \
  -DCMAKE_PREFIX_PATH="/path/to/geant4;/path/to/vecgeom" \
  -DCELERITAS_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCELERITAS_BUILD_TESTS=OFF
cmake --build build && cmake --install build
```

Check out a release tag rather than `develop` for production use.

## Key options

| Option | Values | Notes |
|--------|--------|-------|
| `CELERITAS_USE_<Pkg>` | `ON`/`OFF` | `CUDA`, `HIP`, `Geant4`, `VecGeom`, `ROOT`, `HepMC3`, `MPI`, `OpenMP`, `DD4hep`, `LArSoft`, `covfie`, `PNG`, `Perfetto`. Defaults are detected from the environment, and configure fails if an enabled package is missing. |
| `CELERITAS_BUILTIN_<Pkg>` | `ON`/`OFF` | Use an internally downloaded copy (CLI11, nlohmann_json, GoogleTest, covfie, G4VG, Perfetto) instead of an external one. Downloads need network access to GitHub. |
| `CELERITAS_CORE_GEO` | `VecGeom`, `ORANGE`, `Geant4` | Default: VecGeom if enabled (and not HIP), else ORANGE. `Geant4` is CPU-only (no CUDA/HIP/OpenMP). |
| `CELERITAS_UNITS` | `CGS`, `CLHEP`, `SI` | `CLHEP` matches Geant4 units (Spack default) |
| `CELERITAS_DEBUG` | `ON`/`OFF` | Runtime assertions; slow |
| `CELERITAS_OPENMP` | `disabled`, `event`, `track` | Leave `disabled` for multithreaded Geant4 |
| `CELERITAS_REAL_TYPE` | `double`, `float` | `float` is experimental, not with VecGeom |
| `CMAKE_CUDA_ARCHITECTURES` / `CMAKE_HIP_ARCHITECTURES` | e.g. `80`, `gfx90a` | Exported to downstream projects as their default |
| `BUILD_SHARED_LIBS` | `ON`/`OFF` | Shared libraries are typical for frameworks and plugins |

## Presets

`CMakePresets.json` provides generic presets: `minimal` (fewest
dependencies), `default` (detect from the environment), and `full`. Two
deployment presets are release builds for specific consumers: `larsoft` (DUNE
analysis with LArSoft) and `lhc` (LHC framework integration). Use them with
`cmake --preset=<name>` from the source directory (CMake >= 3.21). Read the
preset definitions in the checked-out `CMakePresets.json` before using them,
since they set options and may expect certain dependencies. Write a
`CMakeUserPresets.json` for site-specific settings instead of long command
lines.

## Common configure problems

- Configure enables `CELERITAS_USE_<Pkg>` automatically when the package is
  found. If that picks up the wrong Geant4 or ROOT, set the option explicitly
  and fix `CMAKE_PREFIX_PATH` order.
- "VecGeom core geometry is incompatible with HIP" or "Geant4 core geometry
  is incompatible with HIP, CUDA, and OpenMP": choose another
  `CELERITAS_CORE_GEO`.
- Errors fetching CLI11 or nlohmann_json on offline machines: install them
  (e.g. via Spack) and set `CELERITAS_BUILTIN_<Pkg>=OFF`.
- VecGeom must be built with CUDA when Celeritas is ("CUDA mismatch between
  the VecGeom installation" in downstream projects).
