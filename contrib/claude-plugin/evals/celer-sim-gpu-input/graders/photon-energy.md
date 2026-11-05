---
type: llm
---

The answer must contain a complete `celer-sim` JSON input. PASS if:
- primaries are photons (PDG 22), one per event, 100 events;
- the energy is 1 GeV expressed in MeV (1000);
- the direction is +z and the position is the origin, using the `_type`
  variant encoding (e.g. `"_type": "delta"`);
- the geometry is `detector.gdml` under `problem.model.geometry`;
- a uniform field of 2 (tesla) along z is set in `problem.field`;
- the GPU is enabled with `"system": {"device": {...}}`.

FAIL if any item is missing, uses wrong units (e.g. 1 for 1 GeV), or puts
keys in the wrong place.
