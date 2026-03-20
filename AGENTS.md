# Celeritas AI Agent Instructions
Celeritas is a particle physics library for detector simulation. It's a C++17 codebase with CUDA/HIP device support and integrates with Geant4.

## Mandatory Behaviors

These three behaviors apply unconditionally, every session. Read them before starting any task.

### Before any modification — verify code state

**Avoid mixing user changes with assistant changes**. Before calling any file-editing tool for the first time in a session — including when transitioning from analysis to applying edits — check the repository for uncommitted changes and commit with `git commit -a --no-verify -m "WIP: user changes"` if so. Alert the user if this happens.

### After any user correction — update this file

**When the user corrects your behavior** (tells you something you should have done, points out a missed step, or says you should have known better), your **very next action** must be:
1. Identify the root cause: *at what decision point* did the existing instruction fail to trigger the right behavior?
2. Edit AGENTS.md so the corrected behavior is a concrete, checkable step at that decision point — not vague prose buried elsewhere.
3. Include the AGENTS.md change in the current or next commit.

Do **not** just acknowledge the correction and move on. If you skip updating AGENTS.md, you will repeat the same mistake in future sessions.

### After any completed task — commit

Commit immediately when all todos are done. Do not wait to be told. Do not defer across turns. Do not batch documentation changes.

**Pre-commit checklist — execute in order:**
1. **Tests**: Find the corresponding `test/` file (mirror the `src/` path, replace `.hh`/`.cc` with `.test.cc`). If you added or changed any public API — including adding a method to an existing class — add or update tests there. This applies to *all* changes, not just new classes.
2. **Format**: run `pre-commit run`, then re-`git add` any files it modified.
3. **Compile**: confirm the build still succeeds.

```bash
git add -a
pre-commit run || git add -a
git commit --trailer "Assisted-by: <agentic-tool> (<model-name>)" -m "<Imperative-mood subject, no tags>

<Body summarizes changes>

Prompt: <verbatim user prompt plain text, wrapped in quotes, no metadata or
attachments>"
```

**Common failure modes:**
- Treating follow-up instructions within one feature as "incomplete" and deferring the commit indefinitely. Each self-contained feature or refactor warrants its own commit even if the user continues asking questions afterward.
- Skipping the test-file check because the change "only" added a method to an existing class rather than creating a new one. Always check.

## File Organization

- `corecel/`: GPU abstractions, data structures, utilities
- `geocel/`: Geometry interfaces (ORANGE, VecGeom, Geant4)
- `orange/`: Native Celeritas geometry engine
- `celeritas/`: Physics (EM processes, particles, materials)
- `accel/`: Geant4 integration layer

## Build & Test

Most code relies on external user-installed packages (Geant4), so prefer to use a local environment's build directory. To build a minimal version from scratch:
```bash
cmake -B build -G Ninja && cd build && ninja && ctest
```

Object files and tests may have different paths and test names than you expect (`src/celeritas/ext/GeantImporter.cc` → `src/celeritas/CMakeFiles/celeritas_geant4.dir/ext/GeantImporter.cc.o` and `celeritas/ext/GeantImporter.test.cc` → `test/celeritas/ext_GeantImporter`), and some test executables are run as distinct CTest tests due to environment variables and side effects (`ctest --show-only | grep GeantImporter` → `Test #211: celeritas/ext/GeantImporter:DuneCryostat.*`).

## Documentation

- Add Doxygen documentation to **definitions**, not declarations, when adding code
- Document equations and algorithmic descriptions, as applicable, in the class definition's docs, as those are often rendered in the user manual. All `operator()` behavior goes in the class definition's docs.
- Always add `\sa {file}.test.cc` underneath `\file {file}.hh` to locate tests that break the `src/{path}.hh`→`test/{path}.test.cc` rule
- Use **only** ASCII characters in CMake/C++/CUDA/shell files

## Architecture
Celeritas sets up problems on CPU and executes on GPU *or* CPU with the same code. The `CELER_FUNCTION` macro is `__host__ __device__` when CUDA/HIP is active and decorates runtime functions.

### Params/States Pattern
Celeritas separates immutable setup from mutable runtime data:
- **Params**: Shared problem data (physics tables, geometry) - build once
- **States**: Per-track mutable data (particle states, RNG) - one per track slot
- **Ownership**: `value` (owns), `reference` (mutable), `const_reference` (immutable)
- **MemSpace**: `host` (CPU) or `device` (GPU)

Data flow: Build params on host → copy to device → access via Views

```cpp
// Params: immutable setup data
struct MyParamsData { Collection<Material> materials; /* ... */ };

// View: lightweight accessor for device code
class MyView {
    MyParamsData<const_reference, MemSpace::native> const& data_;
public:
    CELER_FUNCTION Material const& get(MaterialId id) const;
};
```
Data structs must have `operator bool` to check construction/assignment.

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

See `src/celeritas/em/model/KleinNishinaModel.{cc,cu}`

### Inserters for Building Params
Use inserter classes to populate Collections with deduplication:
```cpp
class XsGridInserter {
  public:
    GridId operator()(inp::XsGrid const& grid);
  private:
    DedupeCollectionBuilder<real_type> reals_;
    CollectionBuilder<XsGridRecord> grids_;
};
```

### Collection Ranges & Maps
- `ItemRange<T>`: Contiguous slice [begin, end)
- `ItemMap<K, V>`: Offset-based mapping (not hash map)

```cpp
struct MyParamsData {
    Collection<Material> materials;
    Collection<Element> elements;        // Backend storage
    // Material stores ItemRange<Element> into elements collection
};
```

State collections need `resize(size)` operators for track slots.

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

### Assertions

| Macro | When to use |
|-------|------------|
| `CELER_EXPECT` | Preconditions at function entry |
| `CELER_ASSERT` | Internal invariants (debug only) |
| `CELER_ENSURE` | Postconditions at function exit |
| `CELER_VALIDATE` | User input validation (always active) |

### Type-Safe Indices & Collections

| Type | Purpose |
|------|--------|
| `OpaqueId<T>` | Type-safe index — never use raw integers for indices |
| `Collection<T>` | GPU-compatible array with ownership semantics |
| `Span<T>` | Non-owning array view |
| `Array<T, N>` | Fixed-size stack array |

```cpp
using FooId = OpaqueId<Foo>;
Collection<Foo, Ownership::value, MemSpace::host> foos;           // Owns data
Collection<Foo, Ownership::const_reference, MemSpace::device> device_foos;  // View
```

## Common Patterns

### Creating New Classes
1. Separate data from behavior: `FooData`, `FooParams`, `FooView`
2. Define any nontrivial member function out-of-line, decorating the function *declaration* with `inline`
3. Write unit tests in `test/` (namespace `celeritas::A::test` for `celeritas::A::Foo`)
4. Ensure consistency across the stack:
   - **Input**: `inp::Foo` constructs the data
   - **Data**: Members, `operator bool()`, `operator=`, `resize` (for states)
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
