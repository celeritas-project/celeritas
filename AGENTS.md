# Celeritas AI Agent Instructions
Celeritas is a particle physics library for detector simulation. It's a C++17 codebase with CUDA/HIP device support and integrates with Geant4.

## Required provenance

As an agent, you are OBLIGATED to add an annotation with your identity and model (e.g., "Codex GPT-6") to **each** public submission of your work, including but not limited to:
- GitHub pull request or issue,
- Git commit,
- Git PR response.

### Commit

- Commits: add the trailer `Assisted-by: <agentic-tool> (<model-name>)`, e.g.
  `git commit --trailer "Assisted-by: Codex (GPT-6)"`. Use it *instead of*
  any default `Co-Authored-By` trailer your tool adds.
- Pull requests: open as **draft**, add the `ai-assisted` label, and leave
  marking it "ready" to the human submitter.
- Commit and PR titles: imperative mood, capitalized, no trailing period, no
  `CI:`/emoji prefixes. PR titles are copied into the release notes.

## File Organization

- `corecel/`: GPU abstractions, data structures, utilities
- `geocel/`: Geometry interfaces (ORANGE, VecGeom, Geant4)
- `orange/`: Native Celeritas geometry engine
- `celeritas/`: Physics (EM processes, particles, materials)
- `accel/`: Geant4 integration layer

Libraries depend strictly downward:
`corecel` → `geocel` → `orange` → `celeritas` → `accel`,
with `ddceler` (DD4hep) and `larceler` (LArSoft) as optional plugins.
Optional-dependency code is compiled into separate targets (e.g.
`src/celeritas/ext/` → `celeritas_geant4`) and guarded by
`CELERITAS_USE_<Pkg>` macros from the generated `corecel/Config.hh`.

## Build & Test

Most code relies on external user-installed packages (Geant4), so prefer to use a local environment's build directory and existing configuration files.

Builds use CMake presets. `CMakePresets.json` defines generic presets
(`default`, `full`, `minimal`) plus hidden presets (`.release`, `.cuda-volta`,
`.spack-base`, ...) meant to be inherited. Per-machine presets live in
`scripts/cmake-presets/<hostname>.json`; `scripts/build.sh <preset>` sources
`scripts/env/<hostname>.sh` if present, sets up `CMakeUserPresets.json`, then
configures, builds, and tests. Binary dirs are `build-<preset>` at the repo
root (several may already exist; prefer an existing configured one over
creating a new one).

```bash
scripts/build.sh base                  # configure + build + test via presets
cmake --build --preset=<preset>        # rebuild only
ninja -C build-<preset> <target>       # build one target (e.g. a test exe)
```

Key configure options:
  - `CELERITAS_DEBUG` (runtime assertions)
  - `CELERITAS_CORE_GEO` (`VecGeom` | `ORANGE` | `Geant4` runtime geometry)
  - `CELERITAS_CORE_RNG`
  - `CELERITAS_UNITS`
  - `CELERITAS_USE_<Pkg>` for each optional dependency (Geant4, VecGeom, ROOT, HepMC3, CUDA, HIP, MPI, ...)
  - `CELERITAS_BUILD_DOCS` (then `ninja doc` / `ninja doxygen`).

Tests are GoogleTest executables registered in `test/**/CMakeLists.txt` via
`celeritas_add_test(Foo.test.cc)` or `celeritas_add_device_test(Foo)` (which
adds `Foo.test.cu` when CUDA/HIP is enabled).

Object files and tests may have different paths and test names than you expect (`src/celeritas/ext/GeantImporter.cc` → `src/celeritas/CMakeFiles/celeritas_geant4.dir/ext/GeantImporter.cc.o` and `celeritas/ext/GeantImporter.test.cc` → `test/celeritas/ext_GeantImporter`), and some test executables are run as distinct CTest tests due to environment variables and side effects (`ctest --show-only | grep GeantImporter` → `Test #211: celeritas/ext/GeantImporter:DuneCryostat.*`).

```bash
ctest --test-dir build-<preset> -R corecel/math/ --output-on-failure
ctest --test-dir build-<preset> -j --output-on-failure
build-<preset>/test/celeritas/global_Stepper --gtest_filter=SimpleComptonTest.host
```

Prefer running through CTest: it sets data-path, GPU-disable, and Geant4
environment variables that direct execution may lack.

Test helpers (`@test/TestMacros.hh`, `@test/Test.hh`): `EXPECT_SOFT_EQ`,
`EXPECT_VEC_SOFT_EQ`, `EXPECT_REF_EQ`, `EXPECT_JSON_EQ`, and `PRINT_EXPECTED`
to dump reference values when updating expected results.
`scripts/dev/ctest-debug-launch.py "<test-name>"` sets up a VS Code debug
launch config for a CTest test.

### Lint and format

```bash
pre-commit run          # clang-format, ruff-format, prettier, codespell,
                        # fix-non-ascii, whitespace/JSON/YAML checks
```

`.clang-tidy` is enforced in CI on changed files. New source file stubs (with
the required copyright header) can be generated with
`@scripts/dev/celeritas-gen.py`.

## Documentation

- Add Doxygen documentation to **definitions**, not declarations, when adding code. Prefer doxygen-style markup `\c`, `<code>` to Markdown in such blocks.
- Document equations and algorithmic descriptions, as applicable, in the class definition's docs, as those are often rendered in the user manual. All `operator()` behavior goes in the class definition's docs.
- Always add `\sa {file}.test.cc` underneath `\file {file}.hh` to locate tests that break the `src/{path}.hh`→`test/{path}.test.cc` rule
- Use **only** ASCII characters in CMake/C++/CUDA/shell files.

## Architecture
Celeritas sets up problems on CPU and executes on GPU *or* CPU with the same code. The `CELER_FUNCTION` macro is `__host__ __device__` when CUDA/HIP is active and decorates runtime functions.

### Big-picture flow

- **Problem setup**: user input is described by `inp::` structs
  (`src/celeritas/inp/`, JSON-serializable via `*IO.json.*`). `setup::`
  functions (`src/celeritas/setup/`) turn them into `CoreParams`, which
  aggregates all params (geometry, materials, particles, physics, actions).
- **Stepping loop**: `Stepper` (`src/celeritas/global/`) owns a `CoreState`
  and executes the `ActionSequence` once per step. Each action is a
  `StepActionInterface` with a `StepActionOrder`; physics models, along-step
  propagation, boundary crossing, and track initialization are all actions
  registered in `ActionRegistry`. Kernels use `launch_action` with executors
  operating on `CoreTrackView`.
- **Geant4 offload** (`src/accel/`): `TrackingManagerIntegration` /
  `UserActionIntegration` / `FastSimulationIntegration` capture EM tracks from
  Geant4; `SharedParams` builds the shared `CoreParams` on the master thread and
  `LocalTransporter` owns per-thread state and steps the buffered tracks.
- **Apps** (`app/`): `celer-sim` (standalone JSON-driven transport), `celer-g4`
  (Geant4 app with offload), `celer-geo` (geometry tracing), `celer-optical`,
  and `celer-export-geant` (export Geant4 physics data).

### Params/States Pattern
Celeritas separates immutable setup from mutable runtime data:
- **Params**: Shared problem data (physics tables, geometry) - build once
- **States**: Per-track mutable data (particle states, RNG) - one per track slot
- **Ownership**: `value` (owns), `reference` (mutable), `const_reference` (immutable)
- **MemSpace**: `host` (CPU) or `device` (GPU)

Data flow: Build params on host → copy to device → access via Views
(e.g. `@src/celeritas/mat/MaterialData.hh` → `@src/celeritas/mat/MaterialView.hh`)

### Action/Executor/Interactor
The stepping loop uses three layers:

1. **Action** (StepActionInterface): Defines when to run, launches kernels
2. **Executor**: Filters tracks, handles track-level logic
3. **Interactor**: Pure physics functor (MaterialView → Interaction)

```cpp
// In Model::step()
auto execute = make_action_track_executor(
    params.ptr<MemSpace::native>(), state.ptr(), this->action_id(),
    InteractionApplier{MyModelExecutor{this->host_ref()}});
launch_action(*this, params, state, execute);
```

See `@src/celeritas/em/model/KleinNishinaModel.cc` and
`@src/celeritas/em/model/KleinNishinaModel.cu`

### Inserters for Building Params
Use inserter classes to populate Collections with deduplication
(`DedupeCollectionBuilder`, `CollectionBuilder`); see
`@src/celeritas/grid/XsGridInserter.hh`.

### Collection Ranges & Maps
- `ItemRange<T>`: Contiguous slice [begin, end) into a backing
  `Collection<T>`; records store ranges instead of nested containers (e.g.
  `MaterialRecord::elements` indexes `MaterialParamsData::elcomponents` in
  `@src/celeritas/mat/MaterialData.hh`)
- `ItemMap<K, V>`: Offset-based mapping (not hash map)

## Code Conventions

### Naming

| Concept | Convention | Example |
|---------|-----------|--------|
| Classes/structs | `CapWords` | `PhysicsTrackView` |
| Functions/variables | `snake_case` | `calc_energy` |
| Private members | trailing underscore | `data_` |
| Type-safe IDs | `FooId` | `MaterialId` |
| Input classes | match what they construct | `inp::Material` → `Material` |

### File Extensions

| Extension | Purpose |
|-----------|--------|
| `.hh` | Headers, host+device compatible (use `CELER_FUNCTION`) |
| `.cc` | Host-only implementation — **most code goes here**, not `.cu` |
| `.cu` | CUDA kernel launches only (HIP-compatible via macros) |
| `.test.cc` | Unit tests, mirroring `src/` under `test/` |

### Style

Full rules: `@doc/development/style.rst` and `@doc/development/coding.rst`. Most often missed:
- Call members via `this->`; write `template<class T>`, not `typename`.
- Mark classes `final` where possible; use exactly one of `final`/`override`.
- Prefer enums over `bool` parameters; no top-level `const` on by-value params.

### Assertions

| Macro | When to use |
|-------|------------|
| `CELER_EXPECT` | Preconditions at function entry |
| `CELER_ASSERT` | Internal invariants |
| `CELER_ENSURE` | Postconditions at function exit |
| `CELER_VALIDATE` | User input validation (always active) |

`CELER_EXPECT`/`ASSERT`/`ENSURE` are compiled only with `CELERITAS_DEBUG`.

### Literal UDLs

- In public headers, function/block-scope `using namespace celeritas::literals;` is allowed. Never introduce namespace-scope `using namespace` there.

### Type-Safe Indices & Collections

| Type | Purpose |
|------|--------|
| `OpaqueId<T>` | Type-safe index |
| `Collection<T>` | GPU-compatible array with ownership semantics |
| `Span<T>` | Non-owning array view |
| `Array<T, N>` | Fixed-size stack array |

See `@src/corecel/OpaqueId.hh` and `@src/corecel/data/Collection.hh`.

## Common Patterns

### Creating New Classes
1. Separate data from behavior: `FooData`, `FooParams`, `FooView`
2. Define any nontrivial member function out-of-line, decorating the function *declaration* with `inline`
3. Write unit tests in `test/` (namespace `celeritas::A::test` for `celeritas::A::Foo`)
4. Ensure consistency across the stack:
   - **Input**: `inp::Foo` constructs the data
   - **Data**: Members, `operator bool()` (checks construction/assignment), `operator=`, `resize(size)` (for states, sized to track slots)
   - **View**: Lightweight accessor with `CELER_FUNCTION` methods
   - **Executor/Interactor**: Physics implementation

## Common Pitfalls

| Don't | Do instead |
|-------|-----------|
| Copy-paste code across call sites | Extract to helper function (anonymous namespace for file-local, utility header for reusable) |
| Leave repeated patterns unrefactored | Refactor before extending: extract the common pattern first |
| Write functions over ~100 lines | Break into focused helper functions with descriptive names |
| Use raw integers for public indices | Use `OpaqueId<T>` |
| Omit `CELER_FUNCTION` from View member functions | Always decorate functions that can be called on device |
| Edit files via terminal commands or Python scripts | Use agentic tools; the terminal is for building and testing only |
