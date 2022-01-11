# Celeritas

The Celeritas project implements HEP detector physics on GPU accelerator
hardware with the ultimate goal of supporting the massive computational
requirements of LHC-HL upgrade.

# Installation and development

This project requires external dependencies to build with full functionality.
However, any combination of these requirements can be omitted to enable
limited development on personal machines with fewer available components.

- [CUDA](https://developer.nvidia.com/cuda-toolkit): on-device computation
- an MPI implementation (such as [Open MPI](https://www.open-mpi.org)): shared-memory parallelism
- [ROOT](https://root.cern): I/O
- [nljson](https://github.com/nlohmann/json): simple text-based I/O for
  diagnostics and program setup
- [VecGeom](https://gitlab.cern.ch/VecGeom/VecGeom): on-device navigation of GDML-defined detector geometry
- [Geant4](https://geant4.web.cern.ch/support/download): preprocessing physics data for a problem input
- [G4EMLOW](https://geant4.web.cern.ch/support/download): EM physics model data
- [HepMC3](http://hepmc.web.cern.ch/hepmc/): Event input
- [SWIG](http://swig.org): limited set of Python wrappers for analyzing input
  data

Build/test dependencies are:

- [CMake](https://cmake.org): build system
- [clang-format](https://clang.llvm.org/docs/ClangFormat.html): formatting enforcement
- [GoogleTest](https://github.com/google/googletest): test harness

## Installing with Spack

[Spack](https://github.com/spack/spack) is an HPC-oriented package manager that
includes numerous scientific packages, including those used in HEP. An included
Spack "environment" (at `scripts/dev/env/celeritas-{platform}.yaml`) defines
the required prerequisites for this project.

The script at `scripts/dev/install-spack.sh` provides a "one-button solution"
to installing and activating the Spack prerequisites for building Celeritas.
Alternatively, you can manually perform the following steps:
- Clone Spack following its [getting started instructions](https://spack.readthedocs.io/en/latest/getting_started.html)
- Add CUDA to your `$SPACK_ROOT/etc/spack/packages.yaml` file
- Run `spack env create celeritas scripts/dev/env/celeritas-linux.yaml` (or
  replace `linux` with `darwin` if running on a mac); then `spack -e
  celeritas concretize` and `spack -e celeritas install`
- Run and add to your startup environment profile `spack env activate
  celeritas`
- Configure Celeritas by creating a build directory and running CMake (or
  `ccmake` for an interactive prompt for configuring options).

An example file for a `packages.yaml` that defines an externally installed CUDA
on a system with an NVIDIA GPU that has architecture capability 3.5 is thus:
```yaml
packages:
  cuda:
    paths:
      cuda@10.2: /usr/local/cuda-10.2
    buildable: False
  all:
    variants: cuda_arch=35
```

## Configuring and building Celeritas

To configure Celeritas, assuming the dependencies you want are located in the
`CMAKE_PREFIX_PATH` search path, and other environment variables such as `CXX`
are set, you should be able to just run CMake and build:
```console
$ mkdir build
$ cd build && cmake ..
$ make
```

Ideally you will build Celeritas with all dependencies to gain the full
functionality of the code, but there are circumstances in which you may not
have all the dependencies or features available. By default, the CMake code in
Celeritas queries available packages and sets several `CELERITAS_USE_{package}`
options based on what it finds, so you have a good chance of successfully
configuring Celeritas on the first go. Two optional components,
`CELERITAS_BUILD_<DEMOS|TESTS>`, will error in the configure if their required
components are missing, but they will update the CMake cache variable so that
the next configure will succeed (but with that component disabled).

For a slightly more advanced but potentially simpler setup, you can use the
CMake presets provided by Celeritas via the `CMakePresets.json` file for CMake
3.21 and higher:
```console
$ cmake --preset=default
```
The three main options are "minimal", "default", and "full", which all set
different expectations for available dependencies.

If you want to add your own set of custom options and flags, create a
`CMakeUserPresets.json` file or, if you are a developer, create a preset at
`scripts/cmake-presets/${HOSTNAME%%.*}.json` and call `scripts/build.sh
{preset}` to create the symlink, configure the preset, build, and test.

## Commit hooks

Run `scripts/dev/install-commit-hooks.sh` to install a git post-commit hook
that will amend each commit with clang-format updates if necessary.

## Contributing

See the [development wiki
page](https://github.com/celeritas-project/celeritas/wiki/Development) for
guidelines and best practices for code in the project.

The [code design
page](https://github.com/celeritas-project/celeritas/wiki/Code-design) outlines
the basic physics design philosophy and classes, and [the layout of some
algorithms and
classes](https://github.com/celeritas-project/celeritas-docs/tree/master/graphs)
are available on the `celeritas-docs` repo.

All submissions to the Celeritas project are automatically licensed under the
terms of [the project copyright](COPYRIGHT) as formalized by the [GitHub terms
of service](https://docs.github.com/en/github/site-policy/github-terms-of-service#6-contributions-under-repository-license).

