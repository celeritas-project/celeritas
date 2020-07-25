# Celeritas

The Celeritas project implements HEP detector physics on GPU accelerator
hardware with the ultimate goal of supporting the massive computational
requirements of LHC-HL upgrade.

# Installation and development

This project requires third-party libraries (TPLs) to build:

- ROOT: for I/O
- an MPI implementation (such as OpenMPI): for shared memory parallelism
- VecGeom: on-device navigation of GDML or ROOT-defined detector geometry
- CUDA: on-device computation.

Development/testing requirements:
- CMake (build system)
- [Git-LFS](https://git-lfs.github.com) (large test files, binary
  documentation)
- LLVM/Clang (formatting enforcement)

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

See the [development wiki page
](https://github.com/celeritas-project/celeritas/wiki/Development) for
guidelines and best practices for code in the project.

The [code design page](https://github.com/celeritas-project/celeritas/wiki/Code-design) outlines the basic physics design philosophy and classes.

Some [codebase images and graphs](https://github.com/celeritas-project/celeritas-docs/tree/master/celeritas-code) are available on the celeritas-docs repo.
