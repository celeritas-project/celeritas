---
name: cmake-link
description: Write or fix the CMake code of a downstream project that uses an installed Celeritas, including find_package, celeritas_target_link_libraries, Celeritas::G4 vs Celeritas::celeritas, CUDA/VecGeom relocatable device code, plugin (MODULE/SHARED) libraries, and .cu sources. Use when editing a CMakeLists.txt that consumes Celeritas or when linking against Celeritas fails.
---

Celeritas exports CMake targets in the `Celeritas::` namespace. The main trap
is CUDA relocatable device code (RDC). When Celeritas is built with **both
CUDA and VecGeom**, device code must be linked in a final step. Plain
`target_link_libraries` then fails or links code that crashes at run time.

## 1. Check the installed configuration

Use the `doctor` facts (run it if not yet done). `celeritas.use` tells you
whether `cuda`/`hip`, `vecgeom`, and `geant4` are enabled. After
`find_package(Celeritas)` the same information is in CMake variables
(`CELERITAS_USE_CUDA`, `CELERITAS_USE_VecGeom`, `CELERITAS_USE_Geant4`,
`CELERITAS_CORE_GEO`, ...).

## 2. Find the package

```cmake
find_package(Celeritas 0.7 REQUIRED)            # minimum version
find_package(Celeritas 0.7 REQUIRED COMPONENTS Geant4)  # fail early if no G4
```

- Component names are the optional package names used by
  `CELERITAS_USE_<Pkg>` (e.g. `Geant4`, `VecGeom`, `CUDA`, `HIP`, `MPI`,
  `HepMC3`, `ROOT`).
- The version file accepts any newer version. While Celeritas is 0.x, asking
  for an older minor version than the one installed prints an
  `AUTHOR_WARNING` ("may not be compatible ... please update your find_package
  version range"). Fix it by raising the minimum or giving a range whose
  maximum covers the installed version (e.g. `0.7...0.8`).
- Users point CMake at the install with `CMAKE_PREFIX_PATH`, `Celeritas_ROOT`,
  or `Celeritas_DIR=<prefix>/lib/cmake/Celeritas`. A Spack-loaded Celeritas
  sets these automatically.
- `find_package(Celeritas)` also finds the dependencies Celeritas was built
  with (Geant4, VecGeom, CUDA toolkit, ...). They must be the same versions;
  a different Geant4 in the environment is a common cause of failures.

## 3. Link

```cmake
add_executable(myapp main.cc)
celeritas_target_link_libraries(myapp PRIVATE Celeritas::G4 ${Geant4_LIBRARIES})
```

- Use `celeritas_target_link_libraries` (and `celeritas_add_library`,
  `celeritas_set_target_properties`, `celeritas_install`, ...) for every
  target that links Celeritas directly *or transitively*. Without RDC they
  forward to the plain CMake commands, so they are always safe.
- `add_executable` needs no wrapper.
- Targets: `Celeritas::G4` is the single Geant4 integration target (an alias
  of `Celeritas::accel`, or of the header-only `Celeritas::BuildFlags` when
  Celeritas has no Geant4). `Celeritas::celeritas` is the core physics
  library, needed directly only when calling non-Geant4 APIs such as custom
  actions. `Celeritas::corecel`, `Celeritas::geocel`, `Celeritas::orange` are
  lower-level libraries.
- Plugin libraries loaded at run time (`dlopen`, framework plugins): prefer
  `celeritas_add_library(myplugin MODULE ...)` so the device link happens in
  the plugin. If the framework forces `SHARED` through its own macros, also
  link the device counterpart of the top-most Celeritas library used
  (`Celeritas::accel_final` for `Celeritas::G4`, `Celeritas::celeritas_final`
  for `Celeritas::celeritas`):

  ```cmake
  target_link_libraries(myplugin
    PRIVATE Celeritas::G4 $<TARGET_NAME_IF_EXISTS:Celeritas::accel_final>
  )
  ```

  `<target>_final` exists only in CUDA RDC builds; the generator expression
  keeps other builds working.

## 4. GPU languages and sources

- In a CUDA build, `CeleritasConfig.cmake` calls `enable_language(CUDA)` only
  if `find_package` runs in the top-level `CMakeLists.txt`. Otherwise it stops
  with "CUDA compiler/linker are required when compiling with CUDA". Fix:
  declare `project(... LANGUAGES CXX CUDA)` or call `enable_language(CUDA)`
  in the top-level `CMakeLists.txt` before `find_package(Celeritas)`. HIP
  works the same way.
- `CMAKE_CUDA_ARCHITECTURES`/`CMAKE_HIP_ARCHITECTURES` default to the values
  Celeritas was built with. Do not override them with incompatible values.
- Only add `.cu` files for user device code (e.g. custom step actions).
  Compile them only when `CELERITAS_USE_CUDA OR CELERITAS_USE_HIP`, as in the
  `polysource_append` macro of
  `example/offload-template/CMakeLists.txt`.
- Guard optional features with the exported variables, e.g.
  `if(NOT CELERITAS_USE_Geant4) message(SEND_ERROR ...)`.

## 5. Verify

Configure and build from a clean build directory, then run the executable
once with `CELER_LOG=debug`. For link errors such as undefined
`__cudaRegisterLinkedBinary*` or `nvlink` symbols, check that *every* target
on the path to the executable uses the `celeritas_` wrappers (or the
`_final` workaround above).

## References

- `doc/usage/integration.rst` (online: https://celeritas-project.github.io/celeritas/user/usage/integration.html)
- `example/data-model/CMakeLists.txt` (core libraries only)
- `example/offload-template/CMakeLists.txt` and `example/geant4/CMakeLists.txt` (Geant4 apps)
- Installed `<prefix>/lib/cmake/Celeritas/CeleritasConfig.cmake` and `CeleritasLibrary.cmake`
