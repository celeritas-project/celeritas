# Celeritas AI Agent Instructions

Celeritas is a GPU-accelerated HEP detector physics library for HL-LHC, integrating with Geant4. It's a C++17 codebase with CUDA/HIP device support.

## Architecture Overview

### Core Components (src/)
- **corecel/**: GPU abstractions, data structures, utilities (host-device compatible code)
- **orange/**: ORANGE geometry engine (native Celeritas geometry)
- **geocel/**: Geometry cell interfaces (ORANGE, VecGeom, Geant4 adapters)
- **celeritas/**: Core physics (EM processes, particles, materials, stepping loop)
- **accel/**: Geant4 integration layer (offload mechanisms, tracking managers)

### Data Flow Pattern
Celeritas separates *shared* data from *state* data:
- **Params**: Immutable problem setup data (physics tables, geometry) - constructed once
- **States**: Mutable per-track data (particle states, RNG states) - one per concurrent event/track
- Data exists in **host** and **device** memory spaces with explicit ownership (`value`, `reference`, `const_reference`)

To support GPU execution, it uses data-oriented design but with object-oriented interfaces (`View`s).

## Build & Test Workflow

### Quick Start

Use a standard cmake workflow:
```bash
cmake --preset=base
cmake --build --preset=base
ctest --preset=base
```

### Testing
- Unit tests in `test/` mirror `src/` structure
- Use `ctest --preset=<preset>-unit` for unit tests, `-app` for app tests
- GPU tests may run serially (controlled by `CELERITAS_TEST_RESOURCE_LOCK`)
- Test naming: `TEST_IF_CELER_DEVICE(name)` for GPU-only, `TEST_IF_CELERITAS_DEBUG(name)` for assertion-dependent tests

## Code Conventions

### File Extensions
- `.hh`: Host-device compatible C++ headers (use `CELER_FUNCTION` macros)
- `.cc`: Host-only C++ (compiled by gcc/clang)
- `.cu`: CUDA kernels and launch code (compiled by nvcc, but should be HIP-compatible via macros)
- `.device.hh/.device.cc`: Requires CUDA/HIP runtime but compilable by host compiler
- `.test.cc`: Unit tests (GoogleTest)

### Host-Device Compatibility
Use these macros (from `corecel/Macros.hh`):
- `CELER_FUNCTION`: Mark functions callable from both host and device
- `CELER_FORCEINLINE_FUNCTION`: Force-inline version
- `CELER_CONSTEXPR_FUNCTION`: Compile-time + host/device
- `CELER_LAUNCH_KERNEL(name, threads, stream, ...)`: Launch device kernels

Example:
```cpp
CELER_FUNCTION real_type calculate(Real3 const& pos) { /* ... */ }
```

### Naming Conventions
- Classes/structs: `CapWords` (e.g., `PhysicsTrackView`)
- Functions/variables: `snake_case`
- Private members: trailing underscore (`data_`)
- Functors: agent nouns (`ModelEvaluator`), instances are verbs (`evaluate_model`)
- OpaqueId types: `FooId` for `Foo` items, `Bar_` tag struct for abstract concepts

### Data Structures
- Use `OpaqueId<T>` for type-safe indices, not raw integers
- End structs with `operator bool()` for validity checks
- Prefer `Collection<T, W, M>` over raw arrays for GPU-compatible storage
- Use `Span<T>` for array views, `Array<T, N>` for fixed-size arrays

### Assertions & Error Handling
- `CELER_EXPECT`: Preconditions (function entry)
- `CELER_ASSERT`: Internal invariants
- `CELER_ENSURE`: Postconditions (function exit)
- `CELER_VALIDATE`: Always-on user input validation
- Debug assertions only active when `CELERITAS_DEBUG=ON`

## Critical Patterns

### Action/Executor/Interactor Paradigm
The stepping loop uses a three-layer pattern:
- **Action**: Implements `StepActionInterface`, defines when to run (`order()`), and launches kernels for host/device via `step()` methods
- **Executor**: Wraps the interactor and handles track-level logic (e.g., `make_action_track_executor` filters by `action_id`)
- **Interactor**: Pure physics functor that operates on a minimal physics information (e.g., `MaterialView`) and returns an `Interaction`

Example flow:
```cpp
// In Model::step() [.cc or .cu]
auto execute = make_action_track_executor(
    params.ptr<MemSpace::native>(), state.ptr(), this->action_id(),
    InteractionApplier{MyModelExecutor{this->host_ref()}});
launch_action(*this, params, state, execute); // or ActionLauncher for device
```

See `src/celeritas/em/model/KleinNishinaModel.{cc,cu}` for a complete example.

### Building Params Data with Inserters
Physics models build device-compatible data using "inserter" classes during construction. Inserters efficiently populate `Collection` objects with deduplication and proper memory layout:
```cpp
class XsGridInserter {
public:
    GridId operator()(inp::XsGrid const& grid); // Returns ID for referencing
private:
    DedupeCollectionBuilder<real_type> reals_;  // Deduplicates identical data
    CollectionBuilder<XsGridRecord> grids_;     // Sequential insertion
};
```
See `src/celeritas/grid/XsGridInserter.hh` and `src/celeritas/em/model/detail/LivermoreXsInserter.hh`.

### Collection Groups: Managing Host-Device Data
Collections manage deeply hierarchical GPU data via type-safe indices and ranges:
- **Collection<T, W, M, I>**: Array-like container with ownership (`value`/`reference`/`const_reference`) and memory space (`host`/`device`)
- **ItemId<T>**: Type-safe index into a `Collection<T>` (actually `OpaqueId<T>`)
- **ItemRange<T>**: Slice of items [begin, end) for variable-length data
- **ItemMap<KeyId, ValueId>**: Maps one ID type to another via offset arithmetic (not a hash map!)

Example data structure:
```cpp
template<Ownership W, MemSpace M>
struct MyParamsData {
    Collection<Material, W, M> materials;       // Array of materials
    Collection<ElementComponent, W, M> components;  // Backend storage
    Collection<double, W, M> reals;             // Backend reals
    // Each Material has ItemRange<ElementComponent> referencing components
    // Templated operator= enables copying across memory spaces
    // Boolean operator validates initialization and copying
};
```

Collections power the params/states architecture: build on host with `Ownership::value`, copy to device, then access via `const_reference` (params) or `reference` (states). See `src/corecel/data/Collection.hh` for details.

### Avoid NVCC When Possible
Most development doesn't involve CUDA code. Kernels (`__global__`) must be in `.cu` files, but CUDA API calls (e.g., `cudaMalloc`) can be in `.cc` files. Include `corecel/DeviceRuntimeApi.hh` for platform-agnostic device APIs. Memory management is implemented with the `Collection` types.

### Test Requirements
Every class needs a unit test with cyclomatic complexity coverage. Detail classes (in `detail/` namespaces) are exempt but still recommended.

## Integration Points

### Geant4 Integration (accel/)
Users integrate via `SharedParams`, `TrackingManagerConstructor`, and run actions (`BeginOfRunAction`, `EndOfRunAction`). See `example/accel/` for templates. Use `celeritas_target_link_libraries()` instead of `target_link_libraries()` to handle VecGeom RDC linking.

### Standalone Execution (app/)
EM-only execution is used for performance testing and verification via `celer-sim`.

### Geometry
Supports ORANGE (native), VecGeom, and Geant4 geometries. GDML is the standard interchange format. Geometry loads through `inp::Model` (see `geocel/inp/Model.hh`).

## Documentation

- Doxygen comments go next to **definitions**, not declarations
- Document `operator()` behavior in class comment, not operator itself
- Use `\citep{author-keyword-year}` for references (maintained in Zotero at `doc/_static/zotero.bib`)
- Physics constants need units and paper citations

## Development Tools

- **pre-commit**: Auto-formats code (clang-format enforces 80-column limit, East const)
- **celeritas-gen.py** (`scripts/dev/`): Generate file skeletons with proper decorations
- **CMake presets**: System-specific configs in `scripts/cmake-presets/<hostname>.json`

## Common Pitfalls

- Never copy-paste code - refactor into reusable functors
- Failing to mark functions `CELER_FUNCTION` will cause "call to __host__ function from __device__" errors

## External Dependencies

Key dependencies (see `CMakeLists.txt` for versions):
- Geant4
- GoogleTest (tests), CLI11 (apps), nlohmann_json (I/O)
- Optional: VecGeom, ROOT, HepMC3, DD4hep, MPI, OpenMP, Perfetto
