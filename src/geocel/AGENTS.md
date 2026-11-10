# Geocel Agent Instructions

`geocel` defines the engine-independent geometry interfaces and types shared by
the ORANGE (`orange/`), VecGeom (`geocel/vg/`), and Geant4 (`geocel/g4/`)
navigation implementations.

## Geometry engine structure

Each engine supplies the same set of pieces:

| Piece | ORANGE | VecGeom | Geant4 |
|---|---|---|---|
| Params (host metadata, `GeoParamsInterface`) | `OrangeParams` | `VecgeomParams` | `GeantGeoParams` |
| Params/state data (`W`, `M` templates) | `OrangeParamsData`, `OrangeStateData` | `VecgeomParamsData`, `VecgeomStateData` | `GeantGeoParamsData`, `GeantGeoStateData` |
| Concrete track view | `OrangeTrackView` | `VecgeomTrackView` | `GeantGeoTrackView` |
| Traits (`GeoTraits<P>`) | `orange/OrangeGeoTraits.hh` | `vg/VecgeomGeoTraits.hh` | `g4/GeantGeoTraits.hh` |

- `GeoTraits<P>` (`@GeoTraits.hh`) maps a params class to its `ParamsData`,
  `StateData`, `TrackView`, and `WrappedTrackView` types, plus feature flags
  (`has_impl_surface`, `has_impl_volume_instance`) and a `name`. Unconfigured
  engines inherit `NotConfiguredGeoTraits`; test with
  `is_geometry_configured_v<P>`.
- Every concrete track view has the same shape: `ParamsRef`/`StateRef` type
  aliases (`NativeCRef<...ParamsData>`, `NativeRef<...StateData>`), a
  `(ParamsRef const&, StateRef const&, TrackSlotId)` constructor, and
  reference members to the params and state refs plus the slot ID. They are
  therefore **cheap to construct but not assignable**: to point a view at
  another slot, construct a new one.
- Track views are `CELER_FUNCTION` (host + device) except for Geant4, which is
  host-only and has stream-local state.

## Track interface layers

The navigation API is documented once, on `GeoTrackInterface<RealType>`
(`@GeoTrackInterface.hh`): initialization via `operator=(GeoTrackInitializer)`,
the `find_next_step` -> `move_to_boundary`/`move_internal` ->
`cross_boundary` sequence, safety-limited moves, and `geo_status()`. Concrete
track views implement the same member functions **non-virtually**. The virtual
interface is reached through adapters:

| Class | Role |
|---|---|
| `GeoTrackInterface<RT>` | Abstract virtual interface (pure virtuals; destructor anchored in `GeoInterface.cc`) |
| `WrappedGeoTrackView<GTV>` (`@WrappedGeoTrackView.hh`) | `final` adapter: concrete view -> virtual interface |
| `CheckedGeoTrackView` (`@CheckedGeoTrackView.hh`) | Decorator around a `unique_ptr<GeoTrackInterface>`: validates calls, counts intersections/safeties, logs (`CELER_LOG_GEO`) |
| `MockGeoTrackView` (`test/geocel/`) | Analytic stand-in used by raytracer tests |

- `WrappedGeoTrackView` stores **copies** of the params/state refs and the
  slot, and holds the concrete view in a `std::optional`. Its move
  constructor re-emplaces the view against the new object's refs (moving a
  view with reference members would otherwise dangle), and
  `set_track_slot(tid)` re-emplaces it in place without reallocating.
- When adding a pure virtual to `GeoTrackInterface`, implement it in **both**
  `WrappedGeoTrackView` and `CheckedGeoTrackView` (the latter inherits from
  the unqualified `GeoTrackInterface<real_type>`, so it is easy to miss with a
  grep). `CheckedGeoTrackView` must reset any per-track cache (e.g.
  `next_step_`) when the track changes.
- Free functions in `@GeoInterface.hh` (`volume_name`, `unique_volume_name`,
  ...) take a `GeoTrackInterface const&` plus params for diagnostics.
- Generic geometry tests (`test/geocel/GenericGeoTestBase.hh`,
  `GenericGeoTestInterface.hh`, `GeoTests.cc`) exercise every engine through
  `WrappedGeoTrackView` and `CheckedGeoTrackView`.

## Core geometry selection (`CELERITAS_CORE_GEO`)

The `celeritas` stepping loop uses a single configure-time engine, selected
by the `CELERITAS_CORE_GEO` CMake option and exposed via the generated
`CELERITAS_CORE_GEO_<VAL>` macros in `corecel/Config.hh`. The selector headers
live in `celeritas/geo/`: `GeoFwd.hh` (type aliases), `CoreGeoParams.hh`,
`CoreGeoTrackView.hh`, `GeoData.hh`, and `CoreGeoTraits.hh`
(`GeoTraits<CoreGeoParams>`).

| Value | Limitation |
|---|---|
| `ORANGE` | Default; always available |
| `VecGeom` | Not with HIP |
| `Geant4` | Not with CUDA, HIP, or OpenMP |
| `dynamic` | Experimental; not with CUDA, HIP, or `CELERITAS_OPENMP=track` |

- `dynamic` uses ORANGE params and data, but `CoreGeoTrackView` is the
  abstract `GeoTrackInterface<real_type>`. Each core (and optical)
  `CoreStateData` owns one `std::shared_ptr` implementation (`geo_track`,
  created in `resize`; see `celeritas/geo/DynamicGeoData.hh`), and the
  reference state holds a raw pointer because state refs must be trivially
  copyable (`CoreState` keeps a `DeviceVector<Ref>` even on host).
  `CoreTrackView::geometry()` calls `set_track_slot` and returns a
  **reference**, so a geometry view of one slot is invalidated by asking for
  another slot of the same state. This is why OpenMP track threading is
  prohibited.
- Bind geometry views in generic code as `auto&& geo = track.geometry();`
  (or `auto const&`), never `auto geo = ...`: copying fails for the abstract
  `dynamic` interface, while a reference binding works for both the by-value
  and by-reference return types.
- Code that constructs a core geometry view directly from geometry
  params/state (rather than via `CoreTrackView`) cannot do so under
  `dynamic`; tests use `test::StandaloneGeoTrackView`
  (`test/celeritas/geo/StandaloneGeoTrackView.hh`) instead.
- Under `dynamic`, `CoreGeoTraits::TrackView` (`OrangeTrackView`) differs from
  `CoreGeoTrackView`: check `std::is_same_v` before using engine-specific
  accessors such as `impl_surface_id`.
- When adding an `#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE` check,
  decide whether it is about ORANGE **data** (also true for `dynamic`) or the
  concrete `OrangeTrackView` (not available to core code under `dynamic`).
