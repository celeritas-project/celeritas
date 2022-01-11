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
`.cuda-volta`, `.spack-base`) useful for inheriting in user presets.

CMake's JSON parser is quite primitive at the moment, and the error messages
it provides are frankly terrible. Use Python's `json` module (or other tools)
to check the syntax of the JSON file, and you can use JSON validators such
as [this one](https://www.jsonschemavalidator.net) to check the schema. Even
with a valid syntax and schema, you might still get an error such as "Invalid
preset" if the value for `configurePreset` is missing or incorrect.

# Development scripts

These scripts are used by developers and by the Celeritas cmake code itself to
generate files, set up development machines, and analyze the codebase.

# Docker scripts

The `docker` subdirectory contains scripts for setting up and maintaining
reproducible environments for building Celeritas and running it under CI.
