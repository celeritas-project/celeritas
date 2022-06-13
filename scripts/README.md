# Build scripts

[CMake
presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) have
replaced the combination of shell scripts and CMake cache setters. Use the
`build.sh` script in this directory to automatically link the
`cmake-presets/${HOSTNAME}.json` file to `${SOURCE}/CMakeUserPresets.json`,
then invoke CMake to configure, build, and test.

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

CMake's JSON parser is quite primitive at the moment, and the error messages
it provides are frankly terrible. Use Python's `json` module (or other tools)
to check the syntax of the JSON file, and you can use JSON validators such
as [this one](https://www.jsonschemavalidator.net) to check the schema. Even
with a valid syntax and schema, you might still get an error such as "Invalid
preset" if the value for `configurePreset` is missing or incorrect.

# Development scripts

These scripts are used by developers and by the Celeritas CMake code itself to
generate files, set up development machines, and analyze the codebase.

# Docker scripts

The `docker` subdirectory contains scripts for setting up and maintaining
reproducible environments for building Celeritas and running it under CI.

# Spack environment

See [the Spack documentation](https://spack.readthedocs.io/en/latest/) for
getting started with Spack. Once it's installed, if you have CUDA and
optionally a CUDA-compatible card on your system, create a YAML configuration
file at
`$SPACK_ROOT/etc/spack/packages.yaml` to inform Spack to build all
supported packages with CUDA and C++17:
```yaml
packages:
  all:
    variants: +cuda cuda_arch=70 cxxstd=17
```
Note that the value of `cuda_arch` is the CUDA GPU architecture and may differ
based on your card's capabilities.

Once Spack is installed, run `spack env create celeritas spack.yaml` and `spack
-e celeritas install` to install a complete environment needed to build
Celeritas with all supported options.
