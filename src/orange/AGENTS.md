# ORANGE Agent Instructions

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

### `GeoStatus` values

| Value | Meaning |
|---|---|
| `interior` | Inside a volume, not on a boundary |
| `exiting_boundary` | On a surface; track is moving outward (default after crossing) |
| `entering_boundary` | On a surface; track is moving inward (set by `move_to_boundary`) |
| `exterior` | Outside the global geometry |
| `error` | Unrecoverable tracking failure; `failed()` returns true |

`flip_boundary` swaps `entering_boundary` ↔ `exiting_boundary` and is called by `set_dir` when a direction change on a surface reverses the crossing sense.

### Typical per-step sequence

```
find_next_step(max_step)   → sets next_step / next_surf / next_univ_level
move_to_boundary()         → physically moves; sets geo_status = entering_boundary
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

**`cross_boundary()`**: flips the sense at the surface level, sets `geo_status = exiting_boundary`, re-initializes the volume via `cross_boundary(local)`, then descends into any daughters below by calling `initialize` at each sub-level.

### Key invariants

- `is_on_boundary()` ≡ `surface_univ_level` is nonzero (the null `UnivLevelId{0}` is the global level, so a nonzero value means a surface is recorded).
- `geo_status == entering_boundary` means the track is pointed back into its current volume; `find_next_step` returns `{0, true}` and `cross_boundary` is a no-op.
- On error, `geo_status = error` is preserved through the end of initialization and boundary crossing so `failed()` stays true.
