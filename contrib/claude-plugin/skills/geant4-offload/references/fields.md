# Magnetic fields: `opts.make_along_step`

Celeritas does not read the Geant4 field manager automatically for uniform
fields. The along-step factory must describe the **same** field the Geant4
application uses, or charged-particle results will differ. Factories and
helpers are declared in the installed `accel/AlongStepFactory.hh`,
`accel/CartMapMagneticField.hh`, `accel/CylMapMagneticField.hh`, and
`accel/RZMapMagneticField.hh`. The field input structs are in
`celeritas/inp/Field.hh`.

## Zero field

```cpp
opts.make_along_step = celeritas::UniformAlongStepFactory();
```

## Uniform field

The factory takes a function, evaluated lazily at setup, that returns an
`inp::UniformField`. Strength units default to tesla (`UnitSystem::si`):

```cpp
opts.make_along_step = celeritas::UniformAlongStepFactory([] {
    celeritas::inp::UniformField field;
    field.strength = {0, 0, 3.8};  // [T]
    return field;
});
```

To restrict the field to some volumes, set `field.volumes` to a set of
logical volume pointers or names. The Geant4 side must match: check the
user's `G4UniformMagField` or field setup (often in
`ConstructSDandField`) for the value and direction.

## Field maps

| Geometry of the map | Factory | Input helper |
|---------------------|---------|--------------|
| 2D r-z | `celeritas::RZMapFieldAlongStepFactory` | Function returning `RZMapFieldInput` (grid sizes, ranges, flattened `field_z`/`field_r`). It can be read from JSON (`celeritas/field/RZMapFieldInputIO.json.hh`). |
| 3D r-phi-z | `celeritas::CylMapFieldAlongStepFactory` | `celeritas::MakeCylMapFieldInput(r_grid, phi_values, z_grid)` samples the Geant4 global field on the given grid (phi in radians). |
| 3D x-y-z | `celeritas::CartMapFieldAlongStepFactory` | `celeritas::MakeCartMapFieldInput(CartMapFieldGridParams)` samples the Geant4 global field on a uniform grid. Requires a Celeritas build with `covfie` (check `celeritas.use` from `doctor`). |

The helpers that sample "the global field" fail with "no Geant4 global field
has been set: cannot build magnetic field map" if no field is attached to the
global `G4FieldManager` when the factory runs. Wrap the call in the factory's
lazy function (as above) so it runs at Celeritas setup, after Geant4 builds
the detector and field.

The sampled grid must cover the tracking volume and be fine enough for the
field gradients. Ask the user for the map's native grid instead of guessing.

## Accuracy and performance knobs

- `driver_options` inside each field input controls the integrator
  tolerances (`FieldDriverOptions`). Match Geant4's `DeltaChord`/`DeltaOneStep`
  intent rather than copying numbers blindly.
- `opts.max_field_substeps` limits substeps per step (default 10).

## Geant4-side adapters

`accel/*MagneticField.hh` also provides Geant4 `G4MagneticField` adapters
(e.g. `celeritas::CylMapMagneticField`). Pure Geant4 runs can use them to get
exactly the same field representation as Celeritas, which removes one source
of discrepancy during validation.
