---
name: build-local
description: Build and test Celeritas using the checkout's existing CMakeUserPresets.json and local development environment. Use when asked to build, compile, configure, or verify the repository with local settings. Overrides CMake Tools.
---

Build Celeritas with the repository's documented development workflow and the user's existing local configuration. `$ARGUMENTS` may name a configure/build/test preset; otherwise use the local development preset described below.

## 1. Identify the local preset

Never read or print `CMakeUserPresets.json` or other preset JSON files directly; they can contain machine-specific paths and settings. Do not edit, replace, unlink, or regenerate the user's preset file.

Use the repository wrapper to have the configured CMake print the available presets:

```sh
./scripts/build.sh --list-presets
```

The wrapper loads the host environment before invoking CMake, which may select the required CMake executable. Its list mode can exit with status 2 even when listing succeeded; treat the printed CMake output as authoritative. For this checkout, use `base` when it appears in the list. If the user names another preset, use it only if CMake lists it. If no suitable development preset is listed or the output is ambiguous, ask the user which preset to use; do not guess or inspect JSON files.

## 2. Configure, build, and test

From the repository root, run the documented wrapper with the selected preset:

```sh
./scripts/build.sh base
```

Replace `base` with the selected preset when appropriate. The wrapper loads the host environment, preserves an existing regular user preset file, configures CMake, builds, and runs CTest with the preset's settings. It may also install the repository's pre-commit hook after a successful build; do not bypass or undo that behavior without the user's direction.

Do not pass build-target options to this wrapper: extra arguments are forwarded to CMake configure. For a target-specific build after configuration, use `cmake --build --preset <preset> --target <target>` and report that CTest was not run unless you run it separately.

## 3. Report

Summarize the preset used, whether configuration and compilation succeeded, and the CTest pass/fail result. If any step fails, report the relevant error and stop before changing local presets, dependency settings, or environment configuration. Never include machine-specific values from the user's preset unless needed to explain a failure and the user has asked for them.
