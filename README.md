# Celeritas

The Celeritas project plans to implement certain HEP detector physics on GPU
accelerator hardware.

# Installation and development

This project requires third-party libraries (TPLs) to build:

- ROOT: for I/O
- an MPI implementation (such as OpenMPI): for shared memory parallelism
- VecGeom: on-device navigation of GDML or ROOT-defined detector geometry
- CUDA: on-device computation.

The CMake build system is also required, and LLVM is required for development
to enforce clang-format.

## Installing with Spack

[Spack](https://github.com/spack/spack) is an HPC-oriented package manager that
includes numerous scientific packages, including those used in HEP. An included
spack "environment" (at `scripts/dev/env/celeritas-{platform}.yaml`) defines
the required prerequisites for this project. A script at
`scripts/dev/install-spack.sh` does roughly the following:
- Clone spack into `$SPACK_ROOT`
- Define a `packages.yaml` file that points to CUDA if available
- Define a `modules.yaml` file to make modules more readable
- Create, concretize, and install the appropriate Spack environment
- Update your `~/.bashrc` file to point to the installed files

## Commit hooks

Run `scripts/dev/install-commit-hooks.sh` to install a git post-commit hook
that will amend each commit with clang-format updates if necessary.

## More details

See the `doc/development.rst` document for guidelines and best practices for
the project.

