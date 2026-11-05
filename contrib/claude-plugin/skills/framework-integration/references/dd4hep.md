# DD4hep / DDG4 (`ddsim`)

Celeritas ships a DDG4 physics-list plugin, `celeritas::dd::CelerPhysics`
(`src/ddceler/CelerPhysics.hh`). It is built when Celeritas is configured
with DD4hep (`CELERITAS_USE_DD4hep`). The plugin registers the Celeritas
tracking manager on the DDG4 physics list and calls `SetOptions` itself, so no
C++ is needed in the experiment's code.

## Steering file

Adopt the plugin into the physics list in the `ddsim` steering file. This
follows `example/ddceler/input/steeringFile.py`:

```python
def setup_physics(kernel):
    from DDG4 import Geant4, PhysicsList

    phys = Geant4(kernel).setupPhysics("QGSP_BERT")
    celer_phys = PhysicsList(kernel, str("CelerPhysics"))
    celer_phys.MaxNumTracks = 2048          # track slots per thread
    celer_phys.InitCapacity = 245760        # initializer capacity per thread
    celer_phys.IgnoreProcesses = ["CoulombScat"]
    # Optional covfie field map instead of the DD4hep ConstantField:
    # celer_phys.FieldMapFile = "path/to/field-map.covfie"
    # celer_phys.FieldMapType = "cartesian"   # or "rz"
    phys.adopt(celer_phys)

runner.physics.setupUserPhysics(setup_physics)
```

| Property | Maps to | Notes |
|----------|---------|-------|
| `MaxNumTracks` | `opts.max_num_tracks` | Required, positive. CPU: thousands. GPU: much larger. |
| `InitCapacity` | `opts.initializer_capacity` | Required, positive |
| `IgnoreProcesses` | `opts.ignore_processes` | List of Geant4 process names |
| `FieldMapFile` | map field along-step | Requires a covfie-enabled Celeritas build |
| `FieldMapType` | `"cartesian"` (default) or `"rz"` | Map layout of `FieldMapFile` |

Behavior to know (read `src/ddceler/CelerPhysics.cc` at the installed
version for the authoritative list):

- Without `FieldMapFile`, the plugin builds a uniform field from the DD4hep
  field description. It accepts only `ConstantField` magnetic components and
  rejects electric fields. Non-constant fields need a covfie map.
- Field driver tolerances come from the steering file's `runner.field.*`
  settings (the DDG4 `MagFieldTrackingSetup` action), so Geant4 and Celeritas
  use the same values. A warning means that action was not found and
  defaults are used.
- Zero-deposition steps are also passed to SDs
  (`opts.sd.ignore_zero_deposition = false`).
- Diagnostics are written to `ddceler.out.json` and the geometry to
  `ddceler.out.gdml` in the working directory.

## Runtime environment

- The plugin library and its `.components` file are installed in Celeritas'
  library directory. Add `<prefix>/lib` to `LD_LIBRARY_PATH`
  (`DYLD_LIBRARY_PATH` plus `DD4HEP_LIBRARY_PATH` on macOS) so the DD4hep
  plugin manager finds `CelerPhysics`.
- Load the DD4hep environment (`DD4hepINSTALL`, its `PYTHONPATH`, and
  ROOT's) as usual.
- `example/ddceler/run-preshower.sh` shows a complete run script that
  switches between Celeritas and pure Geant4 with `CELER_DISABLE=1`.

## Validation

Run the same steering file with and without `CELER_DISABLE=1`
(`run-preshower.sh geant4 ...` vs `run-preshower.sh celeritas ...`) and
compare the hit collections in the output ROOT files per subdetector.
