# ORANGE Agent Instructions

## Geant4-to-ORANGE Conversion (`g4org` namespace)

The `g4org` sub-directory converts an in-memory Geant4 geometry into an `OrangeInput` that is then used to construct `OrangeParams`. The conversion is driven by `Converter`, which takes a `GeantGeoParams` and `VolumeParams` and returns a `Converter::result_type` containing the completed `OrangeInput`.

### Conversion class hierarchy

| Class | Responsibility |
|---|---|
| `Converter` | Top-level entry point; owns and sequences the sub-converters |
| `PhysicalVolumeConverter` | Recursively walks `G4VPhysicalVolume` tree; produces `PhysicalVolume` structs |
| `LogicalVolumeConverter` | Converts one `G4LogicalVolume` (no daughters) to `LogicalVolume`; caches results |
| `SolidConverter` | Converts a `G4VSolid` to an `orangeinp::ObjectInterface`; caches results |
| `Scaler` | Converts CLHEP/Geant4 length units to Celeritas `real_type` (default: mm) |
| `Transformer` | Converts G4 affine/rotation/translation objects to ORANGE `VariantTransform` (daughter-to-parent convention) |

**Intermediate data structures** (in `g4org/Volume.hh`):
- `LogicalVolume` — ORANGE equivalent of `G4LogicalVolume`: holds an `ObjectInterface` plus child placements. Will be renamed `Volume`.
- `PhysicalVolume` — ORANGE equivalent of `G4VPhysicalVolume`: holds a `VolumeInstanceId`, a `VariantTransform`, and a shared `LogicalVolume`. Will be renamed `VolumeInstance`.

**Conversion note:** Geant4 stores "frame" transforms (parent→daughter), but ORANGE requires daughter→parent. `Transformer` inverts as needed.

### Decomposition into unit protos

The converter decomposes the logical volume graph into subgraphs that each become a CSG unit (`UnitProto`):

- **Leaf volumes** (no children) are inlined as material placements into the parent unit.
- **Singly-placed volumes** without transforms are usually also inlined (controlled by `inp::InlineSingletons`).
- **Multiply-placed or rotated volumes** become their own unit proto.

Each `UnitProto` is built with a `build` call that produces surfaces, a CSG tree, and bounding boxes. The proto is then inserted by `UnitInserter` during geometry flattening into the final `OrangeInput`.

### Volume mapping (touchable history)

ORANGE must maintain a bijective map from each implementation volume to the Geant4 touchable hierarchy (unique volume instance). This is achieved in four phases:

1. **Proto conversion**: each child placement records its *proto material ID* (parent volume after subtracting children) in the unit proto input.
2. **Proto build**: proto material IDs are remapped to canonical `VolumeId`s; the result is stored as a *local parent map*.
3. **Geometry flattening** (`UnitInserter`): produces per-universe *local parent volume* and *local volume level* vectors.
4. **Runtime traversal**: the full touchable path is reconstructed by summing local volume levels across the universe stack and following parent links backward.

An ORANGE track state is a `[{universe, impl_volume}]` stack. The *local volume level* values at each universe level sum to the canonical depth in the Geant4 hierarchy.

### Configuration

`inp::OrangeGeoFromGeant` (deserialized from the `G4ORG_OPTIONS` environment variable for debugging) controls:
- `unit_length` and `tol`: length scale and tracking tolerance.
- `inp::InlineSingletons` enum (`none` / `untransformed` / `unrotated` / `all`): how aggressively to inline single-use volumes into their parent universe.

## ORANGE Construction (`orangeinp` namespace)

ORANGE geometry can also be constructed directly (for tests, SCALE import, and manual setups) using the `orangeinp` API before any `OrangeParams` is built.

### Primitives

`IntersectRegionInterface` is the lowest-level primitive: a CSG intersection of half-spaces built from quadric and planar surfaces. Concrete implementations include `Box`, `Cone`, `Cylinder`, `Sphere`, `Ellipsoid`, `Prism`, `GenPrism`, `Paraboloid`, `Hyperboloid`, `Involute`, `ExtrudedPolygon`, and wedge types (`InfAziWedge`, `InfPolarWedge`).

### Object types

`ObjectInterface` implementations compose intersect regions into more complex CSG objects:

| Class | Description |
|---|---|
| `Shape` | Single intersect region; simple convex or quasi-convex volumetric primitive |
| `Solid` | Hollowed/sliced shape (CSG subtraction of two same-type shapes) |
| `PolyCone` / `RevolvedPolygon` / `StackedExtrudedPolygon` | Swept or stacked variants |
| `Transformed` | Applies a rotation/translation to any `ObjectInterface` |
| `JoinObjects` (`AnyObjects` / `AllObjects`) | CSG union or intersection of a list of objects |
| `NegatedObject` | CSG complement |

Objects are held as `shared_ptr<ObjectInterface const>` and can be reused in multiple placements.

### Unit proto and build pipeline

`UnitProto` (a "proto-universe") collects `ObjectInterface` placements, background volumes, and daughter universe references. Its `build` call:

1. Transforms and normalizes surfaces (e.g., flips planes to positive orientation, collapses axis-aligned planes).
2. De-duplicates nearly-coincident surfaces to prevent boundary-crossing errors.
3. Constructs bounding boxes from convex region hints and simplified surfaces.
4. Builds the CSG tree with leaf → interior → root nodes.
5. Simplifies the CSG tree using boundary conditions.

The `UnitInserter` then flattens the `UnitProto` hierarchy into the flat `OrangeInput` arrays consumed by `OrangeParams`.

### Acceleration structure

At runtime, ORANGE uses a **bounding interval hierarchy (BIH)** to accelerate surface intersection queries, avoiding a brute-force search over all surfaces in a unit.

## Runtime Tracking System

### State layout (`OrangeStateData`)

State is split into two tiers:

**Per-track scalars** (indexed by `TrackSlotId`):
- `univ_level` — current deepest `UnivLevelId` the track occupies
- `surface_univ_level` — `UnivLevelId` that owns the current surface (nonzero when on a boundary)
- `surf` — `LocalSurfaceId` at `surface_univ_level`
- `sense` — `Sense` (inside/outside) relative to `surf`
- `geo_status` — `GeoStatus` enum (see below)
- `next_step`, `next_univ_level`, `next_surf`, `next_sense` — lookahead from `find_next_step`

**Per-(track, universe-level)** (2D, flattened; accessed via `LevelStateAccessor`):
- `pos`, `dir` — local position and direction at each universe level
- `vol` — `LocalVolumeId` at each level
- `univ` — `UnivId` at each level

`LevelStateAccessor` (LSA) is the lightweight accessor for the 2D fields. Use `make_lsa()` for the deepest level and `make_lsa(ulev_id)` for a specific level.

### `GeoStatus` values in ORANGE

| Value | Meaning |
|---|---|
| `interior` | Inside a volume, not on a surface |
| `boundary_out` | On a surface; track is moving away from surface (default after crossing) |
| `boundary_inc` | On a surface; track is moving toward surface (set by `move_to_boundary`) |
| `error` | Unrecoverable tracking failure |

`flip_boundary` swaps `boundary_inc` ↔ `boundary_out` and is called by `set_dir` when a direction change on a surface reverses the crossing sense.

### Typical per-step sequence

```
find_next_step(max_step)   → sets next_step / next_surf / next_univ_level
move_to_boundary()         → physically moves; sets geo_status = boundary_inc
cross_boundary()           → flips sense, re-initializes volume at surface level
                             and re-descends into daughters below
```
Or for a step that does not reach a boundary:
```
find_next_step(max_step)
move_internal(dist)        → physically moves; subtracts dist from next_step
```

### Universe hierarchy

ORANGE supports nested universes. The `univ_level` counter (0 = global/world) tracks depth. All per-track loop operations iterate `range(this->univ_level() + 1)`.

`TrackerVisitor` provides type-erased dispatch to `SimpleUnitTracker` (surface-CSG) or `RectArrayTracker`. Key operations:
- `initialize(local)` — find volume for a position/direction
- `intersect(local, max_step)` — distance to next surface
- `cross_boundary(local)` — volume after crossing
- `daughter(vol)` — `DaughterId` if volume contains a nested universe, else null

`TransformVisitor` applies translate/rotate transforms when descending or ascending universe levels. Local positions/directions must be transformed at each level boundary.

### Initialization and boundary crossing

**`operator=(Initializer_t)`** (fresh start): recurses top-down through daughter universes, filling LSA at each level. Sets `geo_status = interior` on success, `geo_status = error` on failure.

**`operator=(DetailedInitializer)`** (secondary/copy): copies all per-track and 2D state from the parent slot, then re-rotates the direction down through all universe levels.

**`cross_boundary()`**: flips the sense at the surface level, sets `geo_status = boundary_out`, re-initializes the volume via `cross_boundary(local)`, then descends into any daughters below by calling `initialize` at each sub-level.

### Key invariants

- `is_on_boundary()` ≡ `surface_univ_level` is nonzero (the null `UnivLevelId{0}` is the global level, so a nonzero value means a surface is recorded).
- `geo_status == boundary_inc` means the track is pointed back into its current volume; `find_next_step` returns `{0, true}` and `cross_boundary` is a no-op.
- On error, `geo_status = error` is preserved through the end of initialization and boundary crossing so `failed()` stays true.
