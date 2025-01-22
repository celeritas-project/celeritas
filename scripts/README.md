# Build.sh, cmake-presents, and env

[CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) have
replaced the combination of shell scripts and CMake cache setters. Use the
`build.sh` script in this directory to automatically link the
`cmake-presets/${HOSTNAME}.json` file to `${SOURCE}/CMakeUserPresets.json`,
then invoke CMake to configure, build, and test. The build script also sources
any script at `env/${HOSTNAME}` for HPC systems that require environment
modules to be loaded.

```console
$ ./build.sh base
+ cmake --preset=base
Preset CMake variables:
# <snip>
+ cmake --build --preset=base
# <snip>
+ ctest --preset=base
# <snip>
```

The main `CMakePresets.json` provides not only a handful of user-accessible
presets (default, full, minimal) but also a set of hidden presets (`.ndebug`,
`.cuda-volta`, `.spack-base`) useful for inheriting in user presets. Make sure
to put the overrides *before* the base definition in the `inherits` list.

# CI scripts

These scripts are executed as part of the Continuous Integration testing. It
also includes an example use case for building an application using Celeritas.

# Development scripts

These scripts are used by developers and by the Celeritas CMake code itself to
generate files, set up development machines, and analyze the codebase.

# Docker scripts

The `docker` subdirectory contains scripts for setting up and maintaining
reproducible environments for building Celeritas and running it under CI.

# Spack environment

Once Spack is installed, run `spack env create celeritas spack.yaml` and `spack
-e celeritas install` to install a complete environment needed to build
Celeritas with all supported options.
