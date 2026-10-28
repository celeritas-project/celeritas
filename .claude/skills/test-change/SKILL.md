---
name: test-change
description: Build and run the unit tests covering changed Celeritas source files. Use after editing code under src/, app/, or test/ to verify the change, or when asked to run tests for specific files.
---

Verify a change by building and running only the affected tests.

## 1. Pick a build directory

Use an existing configured `build-*` directory at the repo root (the one the
user named, otherwise the most recently built). Do not configure a new one.

Always build through `cmake --build`: it uses the `ninja` recorded in
`CMakeCache.txt`, which is often not on `PATH` (e.g. a Spack view).

## 2. Map changed files to tests

Changed files: `$ARGUMENTS` if given, else `git status --porcelain` plus
`git diff --name-only develop...HEAD`.

- `test/<pkg>/<dir>/Foo.test.cc` is itself the test.
- `src/<pkg>/<dir>/Foo.{hh,cc,cu}`: look for `test/<pkg>/<dir>/Foo.test.cc`.
  If absent, check the header's `\sa` lines below `\file`, then
  `grep -rl '<pkg>/<dir>/Foo.hh' test/` for tests that include it.
- Test helpers under `test/` (`*Test.hh`, `*TestBase.*`): find the tests that
  include them.

Name mapping for `test/celeritas/em/KleinNishina.test.cc`:

| What | Name |
|------|------|
| Build target | `celeritas_em_KleinNishina` (`<pkg>_<dir>_<Name>`, `/` -> `_`) |
| Executable | `build-*/test/celeritas/em_KleinNishina` |
| CTest name | `celeritas/em/KleinNishina` |

One executable may be split into several CTest entries with a `:<filter>`
suffix (e.g. `celeritas/ext/GeantImporter:TestEm3*`); match by prefix. Confirm
names with `cmake --build <dir> --target help | grep <Name>` and
`ctest --test-dir <dir> --show-only | grep <Name>`. A test missing from the
list is likely disabled by configure options (e.g. `_needs_double`,
`_needs_geant4_11`); say so rather than forcing it.

## 3. Build and run

```bash
cmake --build build-<preset> --target <target1> <target2> ...
ctest --test-dir build-<preset> -R '^(celeritas/em/KleinNishina)($|:)' --output-on-failure --timeout 60
```

Run through CTest, not the executable directly: CTest sets data paths, GPU
disabling, and Geant4 environment variables.

If library sources changed and no specific test maps to them, run the whole
package directory, e.g. `-R '^celeritas/em/'`.

## 4. Report

List the targets built and the CTest pass/fail summary. For failures, show
the failing assertion output. Do not change expected values to make tests
pass unless the change intentionally alters results; if so, regenerate them
with `PRINT_EXPECTED` and say which values changed and why.
