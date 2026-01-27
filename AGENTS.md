# Celeritas AI Agent Instructions
Celeritas is a GPU-accelerated HEP detector physics library for HL-LHC, integrating with Geant4. It's a C++17 codebase with CUDA/HIP device support.

## Quick Reference

### File Organization
- `corecel/`: GPU abstractions, data structures, utilities
- `geocel/`: Geometry interfaces (ORANGE, VecGeom, Geant4)
- `orange/`: Native Celeritas geometry engine
- `celeritas/`: Physics (EM processes, particles, materials)
- `accel/`: Geant4 integration layer

### Build & Test
```bash
cmake -B build -G Ninja && cd build && ninja && ctest
```

### When to Commit
Commit after major work when:
- New files compile successfully
- Refactored code builds without errors
- Tests pass or aren't affected
- Work is a logical, complete unit

**Commit workflow:**
```bash
git add <files>
pre-commit run        # Auto-formats code
git commit -m "Message" --trailer "Assisted-by: GitHub Copilot (<model-name>)"
```

## Architecture

### Params/States Pattern
Celeritas separates immutable setup from mutable runtime data:
- **Params**: Shared problem data (physics tables, geometry) - build once
- **States**: Per-track mutable data (particle states, RNG) - one per track slot
- **Ownership**: `value` (owns), `reference` (mutable), `const_reference` (immutable)
- **MemSpace**: `host` (CPU) or `device` (GPU)

Data flow: Build params on host → copy to device → access via Views

### Data-Oriented Design with Views
GPU execution requires data-oriented design with object-oriented interfaces:
```cpp
// Params: immutable setup data
struct MyParamsData { Collection<Material> materials; };

// View: lightweight accessor for device code
class MyView {
    MyParamsData<const_reference, MemSpace::native> const& data_;
public:
    CELER_FUNCTION Material const& get(MaterialId id) const;
};
```

## Code Conventions

### Naming
- **Classes/structs**: `CapWords` (e.g., `PhysicsTrackView`)
- **Functions/variables**: `snake_case`
- **Private members**: trailing underscore (`data_`)
- **Type-safe IDs**: `FooId` for `Foo` items
- **Input classes**: Match what they construct (e.g., `inp::Material` → `Material`)

### File Extensions
- `.hh`: Headers (host-device compatible with `CELER_FUNCTION`)
- `.cc`: Host-only implementation
- `.cu`: CUDA kernels (HIP-compatible via macros)
- `.test.cc`: Unit tests (mirror `src/` structure in `test/`)

**Most code is `.cc` - only kernel launches need `.cu`**

### Host-Device Macros
```cpp
CELER_FUNCTION              // Callable from host and device
CELER_FORCEINLINE_FUNCTION  // Force inline + host/device
CELER_CONSTEXPR_FUNCTION    // Compile-time + host/device
```

### Assertions
- `CELER_EXPECT`: Preconditions (function entry)
- `CELER_ASSERT`: Internal invariants (debug only)
- `CELER_ENSURE`: Postconditions (function exit)
- `CELER_VALIDATE`: User input validation (always on)

### Type-Safe Indices & Collections
```cpp
using FooId = OpaqueId<Foo>;
Collection<Foo, Ownership::value, MemSpace::host> foos;  // Owns data
Collection<Foo, Ownership::const_reference, MemSpace::device> device_foos;  // View
```

Key types:
- `Collection<T>`: GPU-compatible array with ownership semantics
- `OpaqueId<T>`: Type-safe index (not raw int)
- `Span<T>`: Non-owning array view
- `Array<T, N>`: Fixed-size stack array

## Critical Patterns

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

## Common Patterns

### Creating New Classes
1. Use `scripts/dev/celeritas-gen.py` for file skeletons
2. Separate data from behavior: `FooData`, `FooParams`, `FooView`
3. Add `operator bool()` validation to data structs
4. Write unit tests in `test/` (namespace `celeritas::A::test` for `celeritas::A::Foo`)

### Consistency Checklist
When adding features, ensure consistency across:
- **Input**: `inp::Foo` constructs the data
- **Data**: Members, `operator bool()`, `operator=`, resize (for states)
- **View**: Lightweight accessor with `CELER_FUNCTION` methods
- **Executor/Interactor**: Physics implementation

## Documentation

- Doxygen comments go on **definitions**, not declarations
- `operator()` behavior documented in class comment
- Citations: `\citep{author-keyword-year}` (refs in `doc/_static/zotero.bib`)
- Physics constants need units and citations

## Development Tools

- `pre-commit`: Auto-formats (clang-format: 80 cols, East const)
- `celeritas-gen.py`: Generate file templates
- `ctest -R <pattern>`: Run specific tests
- `ninja <target>`: Build specific component

## Common Pitfalls

**Don't:**
- Copy-paste code (refactor into reusable functors instead)
- Use raw integers for indices (use `OpaqueId<T>`)
- Forget `CELER_FUNCTION` on device code (→ `__host__/__device__` errors)

**Do:**
- Ensure data consistency (input → data → view → executor)
- Add unit tests for all classes (detail classes excepted)
- Run `pre-commit run` before committing

## External Dependencies

Required: Geant4, GoogleTest (tests), CLI11 (apps), nlohmann_json (I/O)
Optional: VecGeom, ROOT, HepMC3, DD4hep, MPI, OpenMP, Perfetto

See `scripts/spack-packages.yaml` for versions.
