---
name: new-offload-app
description: Start a new standalone Geant4 application with Celeritas offload from the official offload template (detector construction, primary generator, sensitive detector, Celeritas options, CMake), matched to the installed Celeritas version. Use when a user wants a new Geant4+Celeritas project, a minimal reproducer, or a sandbox for trying Celeritas options.
argument-hint: "[target-directory]"
---

The Celeritas source tree contains `example/offload-template`, a small
Geant4 application with Celeritas offload. It is not installed with
Celeritas, so fetch it from GitHub at the **tag that matches the installed
version**. Template APIs track the release (Celeritas 0.6 templates still
use a `RunAction` that 0.7 no longer needs).

## 1. Check prerequisites

Run or recall `doctor`. You need `celeritas.version`, `geant4` in
`celeritas.use`, and a Geant4 that matches. The template requires Geant4 11
or newer (tracking manager offload).

Choose the tag: `v<major>.<minor>.<patch>` from the version, e.g. `v0.7.0`.
For a development version (e.g. `0.7.0-42+abcdef`), use the commit or
`develop`, and tell the user the template may be ahead of their install.

## 2. Fetch only the template

Target directory: `$ARGUMENTS` if given, otherwise ask (default
`./celeritas-offload-app`).

```bash
tag=v0.7.0   # from doctor
git clone --depth 1 --branch "$tag" --filter=blob:none --sparse \
  https://github.com/celeritas-project/celeritas.git celeritas-src
git -C celeritas-src sparse-checkout set example/offload-template
cp -R celeritas-src/example/offload-template <target-directory>
rm -rf celeritas-src
```

If `git` cannot reach GitHub, download the release archive from
https://github.com/celeritas-project/celeritas/releases and copy
`example/offload-template` out of it.

## 3. Adapt

1. In `CMakeLists.txt`:
   - Rename `project(offload-template ...)` and the `run-offload` executable.
   - Delete the `list(APPEND CMAKE_MODULE_PATH ...)` line marked "Only
     necessary for the CI".
   - Keep `celeritas_target_link_libraries` (see `cmake-link`).
2. Keep `find_package(Celeritas <version> REQUIRED)` consistent with the
   installed version.
3. Replace the parts the user cares about, keeping the structure. Paths are
   given as in the Celeritas source tree; in the copy they are under `src/`.
   - `example/offload-template/src/DetectorConstruction.*`: geometry (or
     load GDML) and SD attachment
   - `example/offload-template/src/PrimaryGeneratorAction.*`: primaries
   - `example/offload-template/src/SensitiveDetector.*`: hit processing
     (`ProcessHits` is the only callback Celeritas makes into user code)
   - `example/offload-template/src/MakeCelerOptions.*`: Celeritas options
     (see `geant4-offload`)
4. Optional and experimental: `example/offload-template/src/StepDiagnostic.*`
   shows a GPU-side user action. Remove it (and the `.cu` handling in CMake)
   if not needed. Its API can change between releases.

## 4. Build and run

```bash
cmake -S <target-directory> -B <target-directory>/build -G Ninja
cmake --build <target-directory>/build
cd <target-directory>/build
CELER_LOG=debug ./run-offload ../run.mac
```

Use the renamed executable if step 3 changed it. If Celeritas was built
with MPI and the app is not run under an MPI launcher, also export
`CELER_DISABLE_PARALLEL=1`. Then compare against `CELER_DISABLE=1`, as in
the `geant4-offload` validation steps.

## 5. Report

Tell the user the tag used, the files changed, and how to build and run.
