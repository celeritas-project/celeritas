---
name: configure-local
description: Configure Celeritas using system-specific CMakeUserPresets.json and local development environment. Use when asked to configure or when no build directory is found. Overrides CMake Tools configuration.
---

Configure Celeritas with the repository's documented development workflow and the user's existing local configuration. `$ARGUMENTS` may name a CMake configure preset; this skill only configures and does not accept build targets. With no arguments, use the local development preset described below.

## 1. Identify the local preset

Never read or print `CMakeUserPresets.json` or other preset JSON files directly; they can contain machine-specific paths and settings. Do not edit, replace, unlink, or regenerate the user's preset file yourself.

Use the repository wrapper to have the configured CMake print the available presets:

```sh
<source>/scripts/build.sh --list-presets
```

The wrapper loads the host environment before invoking CMake, which may select the required CMake executable. Note that it runs preset setup before parsing `--list-presets`: it can create and stage a machine-specific preset file, and create or update `CMakeUserPresets.json` (including replacing an existing symlink). This mode is not read-only; explain this side effect and get confirmation if the configure request does not already authorize it. Its list mode will exit if listing succeeded; treat the printed CMake output as authoritative. For this checkout, set `<preset>` to `base` when it appears in the list. If the user names another preset, use it only if CMake lists it. If no suitable development preset is listed or the output is ambiguous, ask the user which preset to use; do not guess or inspect JSON files.

## 2. Configure

Run the documented wrapper with the selected preset:

```sh
<source>/scripts/build.sh --no-build <preset>
```

The wrapper loads the host environment and configures CMake with the user preset settings. It may also install the repository's pre-commit hook after a successful build; do not bypass or undo that behavior without the user's direction. Do not pass build-target options to this wrapper: extra arguments are forwarded to CMake configure.

Carefully examine stdout if the script succeeds (error code 0): the script may print an updated `CMAKE` path or `ENV_SCRIPT` environment to use.

## 3. Check that Geant4 is enabled

After a successful configure, check the cache in the build directory:

```sh
grep -E '^CELERITAS_USE_Geant4:' <build-dir>/CMakeCache.txt
```

If the value is not `ON`, this is a **serious problem** for local development,
even though configuration "succeeded": Geant4-dependent libraries and tests
(Geant4 geometry/GDML loading, ORANGE `g4org` conversion, physics import, most
EM/optical physics tests, and `celer-g4`) are silently disabled, so a passing
test run covers only a fraction of the code. The usual cause is a preset that
was not picked up (e.g., the wrapper's warning that `CMakeUserPresets.json`
does not include the machine-specific preset) or a host environment (e.g.,
Spack view) that was not loaded.

When Geant4 is disabled:
- Put a prominent **WARNING: Geant4 is disabled** at the top of your report,
  with any related wrapper warnings from the configure output.
- Do not proceed to building or testing (including via other skills) until the
  user acknowledges it, unless they already said a Geant4-free build is
  acceptable.
- Never describe test results from such a build as complete: list the
  disabled tests and say they were not run.

## 4. Report

Summarize the preset used, whether configuration succeeded, whether Geant4 is enabled, and whether an environment update is required. If any step fails, report the relevant error and stop before changing local presets, dependency settings, or environment configuration. Never include machine-specific values from the user's preset unless needed to explain a failure and the user has asked for them.
