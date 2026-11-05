---
name: test-change
description: Build and run the unit tests covering changed Celeritas source files. Use after editing code under src/, app/, or test/ to verify the change, or when asked to run tests for specific files.
---

Verify a change by building and running only the affected tests.

## 1. Pick a build directory

Find an existing configured directory (in decreasing preference):
- user-specified if applicable
- `build*` in repo root (most recently built)
- check `.clangd` for `CompileFlags.CompilationDatabase`

If none is available, STOP and ask user for instructions.

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
| Executable | `<build-dir>/test/celeritas/em_KleinNishina` |
| CTest name | `celeritas/em/KleinNishina` |

One executable may be split into several CTest entries with a `:<filter>`
suffix (e.g. `celeritas/ext/GeantImporter:TestEm3*`); match by prefix.
Confirm executable names with `cmake --build <build-dir> --target help | grep <Name>` and
test names with `ctest --test-dir <dir> --show-only | grep <Name>`.
A test missing from the list may be disabled by configure options; say so rather than forcing it.

If the found test for a file does not mirror its source path,
add `\sa <pkg>/<dirs...>/<file>.test.cc` under `\file` in its header.

## 3. Build and run

```bash
cmake --build <build-dir> --target <target1> <target2> ...
ctest --test-dir <build-dir>  -R '^(celeritas/em/KleinNishina)($|:)' --output-on-failure --timeout 60
```

Run through CTest, not the executable directly: CTest sets data paths, GPU
disabling, and Geant4 environment variables.

Changed file names may not directly match build system/target names: e.g.,
`src/celeritas/ext/GeantImporter.cc` → `src/celeritas/CMakeFiles/celeritas_geant4.dir/ext/GeantImporter.cc.o`
and `celeritas/ext/GeantImporter.test.cc` → `test/celeritas/ext_GeantImporter`.

If library sources changed and no specific test maps to them, run the whole
package directory, e.g. `-R '^celeritas/em/'`.

## 4. Report

List the targets built and the CTest pass/fail summary. For failures, show
the failing assertion output. Do not change expected values to make tests
pass unless the change intentionally alters results; if so, regenerate them
with `PRINT_EXPECTED` and say which values changed and why.
