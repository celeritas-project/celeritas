# Library patterns

## Big-picture flow

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

## Patterns

`CELER_FUNCTION` for host/device function decorators.
An old bug in ROCm requires `CELER_CTAD_FUNCTION` for template deduction.

### Params/States
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
`@src/celeritas/em/model/KleinNishinaModel.cu` as an example.

### Type-Safe Indices & Collections
| Type | Purpose |
|------|--------|
| `OpaqueId<T>` | Type-safe index |
| `Collection<T>` | GPU-compatible array with ownership semantics |
| `Span<T>` | Non-owning array view |
| `Array<T, N>` | Fixed-size stack array |

See `@src/corecel/OpaqueId.hh` and `@src/corecel/data/Collection.hh`.

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

See `\page collections` documentation in `@src/corecel/data/Collection.hh` .

### Class groups
1. Separate data from behavior: `FooData`, `FooParams`, `FooView`
2. Define any nontrivial member function out-of-line, decorating the function *declaration* with `inline`
3. Write unit tests in `test/` (namespace `celeritas::A::test` for `celeritas::A::Foo`)
4. Ensure consistency across the stack:
   - **Input**: `inp::Foo` constructs the data
   - **Data**: Members, `operator bool()` (checks construction/assignment), `operator=`, `resize(size)` (for states, sized to track slots)
   - **View**: Lightweight accessor with `CELER_FUNCTION` methods
   - **Executor/Interactor**: Physics implementation

## Precision

`real_type` (`corecel/Types.hh`) is `float` or `double` per
`CELERITAS_REAL_TYPE`; both must build and pass. Patterns:
- Use `real_type`, `Real3` (`Array<real_type, 3>`), and `Quantity` aliases in
  library code; never hard-code `double` except at external-API boundaries.
- External libraries (LArSoft, ROOT, Geant4/CLHEP) are double. Keep their
  converters (e.g. `larceler/Convert.hh`) at the external precision and narrow
  at the call site with `static_array_cast<real_type>(...)`
  (`corecel/cont/Array.hh`). `Array{a, b, c}` deduces its type via CTAD, so
  `static_array_cast<real_type>(Array{x, y, z})` needs no explicit `double`.
- Compare with `SoftEqual` (`corecel/math/SoftEqual.hh`) and test tolerances
  from `test/Test.hh`, not exact equality.
- Tests (`test/TestMacros.hh`): `EXPECT_REAL_EQ`, `TEST_IF_CELERITAS_DOUBLE`
  for double-only tests, and for stochastic or reference-bound results
  `if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE) { exact ref }
  else { smoke check }`. Write the `else` inline rather than hiding it in a
  helper, so failures point at the call site.
- Compiler false positives that appear only in float builds are silenced with
  `celer_silence_flag` in `src/CMakeLists.txt`, not per-file options.
- Build float in a separate build/install dir (preset
  `CELERITAS_REAL_TYPE=float`); `larceler` is covered.

## Style

### Helper functions and classes
- If used only inside a translation unit, declare in an anonymous namespace to
  hide symbol visibility
- If used in a header or reused across files in the same library, put in a
  detail file
- If used across libraries, helper becomes a main library class (and needs tests)

### Literal UDLs
- In public headers, function/block-scope `using namespace celeritas::literals;` is allowed. Never introduce namespace-scope `using namespace` there.

### Assertions
| Macro | When to use |
|-------|------------|
| `CELER_EXPECT` | Preconditions at function entry |
| `CELER_ASSERT` | Internal invariants |
| `CELER_ENSURE` | Postconditions at function exit |
| `CELER_VALIDATE` | User input validation (always active) |

`CELER_EXPECT`/`ASSERT`/`ENSURE` are compiled only with `CELERITAS_DEBUG`.
