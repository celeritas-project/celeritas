---
name: configure-local
description: Configure Celeritas using system-specific CMakeUserPresets.json and local development environment. Use when asked to configure, build, or compile the first time. Overrides CMake Tools configuration.
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

## 3. Report

Summarize the preset used, whether configuration succeeded, and whether an environment update is required. If any step fails, report the relevant error and stop before changing local presets, dependency settings, or environment configuration. Never include machine-specific values from the user's preset unless needed to explain a failure and the user has asked for them.
