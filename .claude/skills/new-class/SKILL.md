---
name: new-class
description: Scaffold new Celeritas source and test files with the required copyright header and register them in CMake. Use when creating any new .hh/.cc/.cu/.test.cc file.
---

Create new files with `scripts/dev/celeritas-gen.py`; never hand-write the
copyright header. `$ARGUMENTS` names the class/files (e.g.
`src/celeritas/em/model/FooModel`).

## 1. Generate stubs

Run from the repo root with repo-relative paths. It skips files that exist.

```bash
python3 scripts/dev/celeritas-gen.py src/<pkg>/<dir>/Foo.hh src/<pkg>/<dir>/Foo.cc
python3 scripts/dev/celeritas-gen.py test/<pkg>/<dir>/Foo.test.cc
```

Only generate what is needed: header-only classes get no `.cc`; add `.cu`
only for kernel launches (`.test.cu` + `.test.hh` only for device tests).

Namespaces: the script infers only `celeritas`, `celeritas::test`, and a
trailing `::detail`. For any other namespace, pass `-n` explicitly, and give
the test the matching `::test` namespace:

```bash
python3 scripts/dev/celeritas-gen.py -n celeritas::optical src/celeritas/optical/Foo.hh
python3 scripts/dev/celeritas-gen.py -n celeritas::optical::test test/celeritas/optical/Foo.test.cc
```

## 2. Register in CMake

- Library `.cc`: add to the `SOURCES` list in `src/<pkg>/CMakeLists.txt`,
  keeping the surrounding order. A `.cc` + `.cu` pair uses
  `celeritas_polysource(<dir>/Foo)` instead. Code depending on an optional
  package (e.g. Geant4 in `ext/`) goes in that package's source list.
- Test: add `celeritas_add_test(<dir>/Foo.test.cc)` (or
  `celeritas_add_device_test(<dir>/Foo)` with a `.test.cu`) to
  `test/<pkg>/CMakeLists.txt` in the matching section, copying any
  `${_needs_double}`-style conditions its neighbors use.
- If the test does not mirror the source path, add `\sa <path>.test.cc`
  under `\file` in the header.

## 3. Fill in

Remove unused template boilerplate, document the class at its definition,
then build and run the new test (the `test-change` skill covers this).
