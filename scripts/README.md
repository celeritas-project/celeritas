# Build.sh, cmake-presents, and env

[CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) have
replaced the combination of shell scripts and CMake cache setters. Use the
`build.sh` script in this directory to automatically set up
`${SOURCE}/CMakeUserPresets.json` from the `cmake-presets/${HOSTNAME}.json`
file, then invoke CMake to configure, build, and test. The build script also
sources any script at `env/${HOSTNAME}` for HPC systems that require
environment modules to be loaded.

If the system presets include the main presets file with
`"include": ["${sourceDir}/CMakePresets.json"]` (which requires presets version
9 and CMake 3.30), `CMakeUserPresets.json` is a small regular file that
includes them. Otherwise it is a symbolic link to them. Only the former is
copied into new Claude Code worktrees by the top-level `.worktreeinclude` file.
The script only ever replaces a symbolic link: if `CMakeUserPresets.json` is a
regular file that does not include the system presets, it warns and leaves the
file alone.

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
presets (default, full, minimal) but also a set of hidden presets (`.release`,
`.cuda-volta`, `.nobuiltin`) useful for inheriting in user presets. Make sure
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

# Spack environments

The `spack` directory has a list of dependency requirements and several
environments for different use cases. The prefixes signify:
- `env`: spack environment (use `spack env create celeritas filename.yaml`)
- `reqs`: requirements for correctly building Celeritas (`reqs-celer`) or the
  cached CI (`reqs-ci`)
- `prefs`: default variants for associated packages
- `ext`: external packages and compilers defined for a particular platform
