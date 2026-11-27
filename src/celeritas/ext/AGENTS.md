# Geant4 import

`GeantImporter` converts the in-memory Geant4 geometry, materials, and physics
into `ImportData` (and the `inp::` structs it holds). Keep each piece of import
code tied to the one kind of Geant4 object it reads.

## Domains

| Class | Reads | Fills |
|---|---|---|
| `GeantImporter` (`import_*` functions) | Material, element, particle, volume tables; EM parameters; `RINDEX` | Top-level `ImportData` members; drives the process loop |
| `detail::GeantPhysicsLoader` | One `G4VProcess` (dispatched by type or name), its settings and `G4OpticalParameters`; simple per-material grids such as `ABSLENGTH` | `optical_physics`, `mucf_physics`, decay, muon pair production |
| `detail::GeantProcessImporter`, `GeantModelImporter` | Legacy EM processes, models, and cross section tables | `ImportProcess` |
| `detail::GeantScintillationLoader` | One material's `G4MaterialPropertiesTable` | `inp::ScintillationMaterial` |
| `detail::GeantSurfacePhysicsLoader` | `G4OpticalSurface` properties | `inp::OpticalSurfacePhysics` |
| `detail::GeantMaterialPropertyGetter`, `GeantOpticalMatHelper` | Property table access with unit conversion | (helpers only) |

## Rules

- Material-property loaders (e.g. `GeantScintillationLoader`) read *only*
  the material properties table. Do not pass them a `G4VProcess` or
  `G4OpticalParameters`. Per-material parsing of more than a few properties
  goes in such a loader rather than in `GeantPhysicsLoader`.
- `GeantPhysicsLoader` owns the link to the Geant4 process. Settings from the
  process (or `G4OpticalParameters`) are applied there, in an explicit
  override step *after* the material data is loaded. Example: scintillation
  rise times are cleared unless `G4Scintillation::GetFiniteRiseTime()`.
- Defaults that Geant4 applies to missing material properties belong in the
  material-property loader, with a comment citing the Geant4 behavior.
- Code must build with all supported Geant4 versions (10.5 and higher: see the
  top-level README, `.github/workflows/build-spack.yml`, or
  `scripts/spack/reqs-celer.yaml`): guard newer APIs with `G4VERSION_NUMBER`.
- Tests: material-property loaders are unit-tested on hand-built
  `G4Material`s under `test/celeritas/ext/detail/`; full import (including
  process overrides) is checked by `test/celeritas/ext/GeantImporter.test.cc`.
