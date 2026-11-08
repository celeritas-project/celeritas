# Installing with Spack

The `celeritas` Spack package (https://packages.spack.io/package.html?name=celeritas)
is the recommended route. Check the variants of the user's Spack version with
`spack info celeritas`, since they change between releases. As of Celeritas
0.7:

| Variant | Default | Meaning |
|---------|---------|---------|
| `+cuda cuda_arch=<cc>` | off | NVIDIA GPU. An architecture is required, e.g. `cuda_arch=80` for A100. |
| `+rocm amdgpu_target=<gfx>` | off | AMD GPU, e.g. `amdgpu_target=gfx90a`. Conflicts with `+cuda` and `+vecgeom`. |
| `+vecgeom` | on | VecGeom geometry. Disable (`~vecgeom`) to use ORANGE. Also disable it for ROCm and for `real_type=float`. |
| `+geant4` | on | Geant4 integration (needed for offload and GDML/physics import) |
| `+hepmc3` | on | HepMC3 event I/O |
| `+root` | off | ROOT I/O (physics export, MC truth, ROOT event files) |
| `+covfie` | on | Map magnetic fields (Cartesian/RZ field maps) |
| `+dd4hep` | off | DD4hep/DDG4 `CelerPhysics` plugin |
| `+openmp` | off | OpenMP. Do not use with multithreaded Geant4 apps. |
| `+debug` | off | Runtime assertions (slow) |
| `units=` | `clhep` | Native unit system (`clhep` = mm/MeV/ns like Geant4, `cgs`, `si`) |
| `real_type=` | `double` | `float` is experimental and incompatible with VecGeom |
| `cxxstd=` | `17` | Must match the C++ standard of Geant4 and the application |

Dependencies with `+cuda` must also be CUDA-enabled with the same
architecture (VecGeom, covfie). The package enforces this.

## Typical specs

```bash
spack install celeritas +cuda cuda_arch=80            # NVIDIA A100, VecGeom
spack install celeritas +rocm amdgpu_target=gfx90a ~vecgeom   # AMD MI250X, ORANGE
spack install celeritas ~vecgeom                      # CPU only, ORANGE
spack install celeritas +cuda cuda_arch=90 +root +dd4hep
spack load celeritas                                  # add to PATH/CMAKE_PREFIX_PATH
```

## Reusing the experiment's software

- Point Spack at existing installations instead of rebuilding them:
  `spack external find cuda`, and declare the experiment's Geant4/ROOT as
  externals in `packages.yaml`, or add `celeritas` to the experiment's own
  Spack environment (e.g. Key4hep).
- To make CUDA the default for all packages:
  `spack config add "packages:all:prefer:cuda_arch=80"` (or the
  `variants` form in older Spack versions).
- After changing defaults, reinstall with `spack install --fresh celeritas`
  so old concretizations are not reused.

## Development-style dependency environments

`scripts/spack/env-*.yaml` in the Celeritas source define environments with
all dependencies (e.g. `env-cuda.yaml`, `env-rocm.yaml`, `env-full.yaml`) and
the minimum versions in `reqs-celer.yaml`. They are useful when building
Celeritas from source on top of Spack-provided dependencies:

```bash
spack env create celeritas scripts/spack/env-cuda.yaml
spack env activate celeritas
spack install
```
