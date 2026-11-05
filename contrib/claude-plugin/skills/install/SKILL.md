---
name: install
description: Choose and carry out a Celeritas installation for downstream use - Spack specs and variants (CUDA/ROCm, VecGeom vs ORANGE, Geant4, ROOT, HepMC3, DD4hep), building from source with CMake options and presets, HPC and container environments, and LArSoft/DUNE (FNAL Spack, Apptainer). Use when a user needs Celeritas installed, rebuilt for a different GPU or geometry, or made consistent with their experiment's software stack.
---

The build configuration fixes GPU support, the geometry engine, the unit
system, and optional I/O, and none of it can be changed at run time. Settle
these choices with the user *before* building. Use the `doctor` skill to
inspect what is installed now and to verify the result.

## 1. Gather requirements

Ask, or infer from the user's environment:

1. **Consumer:** a Geant4 application or framework (needs Geant4, and it must
   be the *same* Geant4 the application uses), or standalone apps only.
2. **Hardware:** NVIDIA GPU (and its compute capability), AMD GPU (its
   `gfx` target), or CPU only.
3. **Geometry:** VecGeom (GDML on CUDA, the default) or ORANGE (required for
   AMD/HIP; Celeritas' native engine). See [references/compat.md](references/compat.md).
4. **Integrations:** DD4hep plugin, LArSoft modules, ROOT output, HepMC3
   input, MPI.
5. **Existing stack:** an experiment Spack environment, CVMFS/LCG views,
   Key4hep, FNAL Spack/UPS, or containers. Reusing the experiment's Geant4 and
   ROOT is usually mandatory.

## 2. Choose the route

| Situation | Route |
|-----------|-------|
| Standard workstation or HPC, Spack available or acceptable | `spack install celeritas ...`: [references/spack.md](references/spack.md) |
| Experiment stack already provides Geant4/ROOT/etc. | Build from source against that stack with CMake: [references/cmake-options.md](references/cmake-options.md) |
| LArSoft/DUNE | FNAL Spack inside the DUNE Apptainer image, `larsoft` CMake preset: see the installation manual's LArSoft section |
| Development of Celeritas itself | Not this plugin's scope. Point to `scripts/build.sh` and the developer docs. |

## 3. Build

Follow the chosen reference. Rules that apply to every route:

- Build Celeritas against exactly the Geant4, VecGeom, ROOT, and DD4hep
  installations the application will use at run time.
- Set the GPU architecture explicitly (`cuda_arch=80`, `CMAKE_CUDA_ARCHITECTURES=80`,
  `amdgpu_target=gfx90a`). A mismatch shows up only at run time.
- Use a release build for production and performance. Debug assertions
  (`+debug`, `CELERITAS_DEBUG=ON`) are for diagnosing problems.
- Keep the install prefix stable: downstream CMake finds Celeritas through
  `CMAKE_PREFIX_PATH`, `Celeritas_ROOT`, or `Celeritas_DIR`.

## 4. Verify

1. Run `doctor` against the new prefix. Check version, `use` list, geometry,
   GPU architectures vs the device, and the Geant4 version match.
2. Run `celer-config --device` (or `celer-sim device`) on a GPU node: it must
   report the GPU.
3. Optionally build a small downstream project against the install (see
   `cmake-link`) or the `new-offload-app` template.

## References

- Installation manual: https://celeritas-project.github.io/celeritas/user/usage/installation.html
- [references/spack.md](references/spack.md), [references/cmake-options.md](references/cmake-options.md), [references/compat.md](references/compat.md)
