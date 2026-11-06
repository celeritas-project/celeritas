---
type: llm
---

PASS if the steering-file code:
- creates the Celeritas DDG4 physics plugin `CelerPhysics` with
  `PhysicsList(kernel, "CelerPhysics")` and adopts it into the physics list
  (e.g. `phys.adopt(celer_phys)` inside a function registered with
  `runner.physics.setupUserPhysics(...)`);
- sets the required `MaxNumTracks` and `InitCapacity` properties; and
- mentions that Celeritas must be built with DD4hep support and its library
  directory must be on the library path so DD4hep can find the plugin.

FAIL if it invents a different plugin name or API, omits the required
properties, or claims C++ changes to the experiment code are required.
