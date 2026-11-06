# JSON encoding of `celeritas::inp`

Summarized from the user manual's JSON input page
(`doc/usage/execution/json-input.rst`). The C++ headers in
`celeritas/inp/` define every field. The `*IO.json.cc` files next to them
show the exact JSON names when in doubt.

## Structs, containers, enums

- A struct is a JSON object with the same member names.
- `std::vector`/`Array` are arrays. `std::unordered_set` is an array of
  unique values. `std::map` is an object.
- Enums are strings, e.g. `"linear"`, `"poly_spline"`, `"cubic_spline"`,
  never integers.

## Variants: the `_type` key

A `std::variant` is an object whose `_type` selects the alternative. The
remaining keys are that alternative's members:

```json
{ "_type": "uniform", "strength": [0.0, 0.0, 1.0] }
```

`_type` strings do not always match C++ type names. Known values:

| Input | `_type` values |
|-------|----------------|
| `problem.field` | `none`, `uniform`, `rzmap`, `cylmap`, `cartmap` |
| `events.generator` | `primary`, `sample`, `read` |
| `physics_import` | `geant`, `file` |
| Distributions (`shape`, `angle`, `energy`) | e.g. `delta` (fixed value), `isotropic` |

For anything else, look for the `json_type_pair("...")` or
`CELER_JSON_LOAD_VARIANT` names in the corresponding `*IO.json.cc` file.

## Optionals: omitted vs `null` vs `{}`

| JSON | Effect on a `std::optional<T>` |
|------|-------------------------------|
| key omitted | Keep the existing default (which may be "unset") |
| `null` | Reset to unset |
| a value | Set to that value |
| `{}` | Set to a default-constructed `T` |

The important case is `"system": {"device": {}}`, which **enables** the GPU
with default settings. Omitting `device` leaves it disabled.

## Units

Each field documents its units in the header comment, e.g. `[MeV]` or
`[len]`:

- Field strength defaults to tesla (`UnitSystem::si`).
- Event generator energies are in MeV.
- `[len]` means the build's native length unit (`celeritas.units` from
  `doctor`; `cgs` means centimeters).

Check the field's comment before converting values from Geant4, whose
internal length unit is mm.

## Example: events

```json
"events": {
  "generator": {
    "_type": "primary",
    "shape": {"_type": "delta", "value": [0.0, 0.0, 0.0]},
    "angle": {"_type": "isotropic"},
    "energy": {"_type": "delta", "value": 10.0},
    "num_events": 1,
    "primaries_per_event": 10000,
    "pdg": [11]
  },
  "merge": false
}
```

Replay a file instead:

```json
"events": {"generator": {"_type": "read", "event_file": "offloaded.jsonl"}}
```

`.jsonl` files need no optional dependencies. `.root` needs ROOT. Other
extensions are read through HepMC3.
