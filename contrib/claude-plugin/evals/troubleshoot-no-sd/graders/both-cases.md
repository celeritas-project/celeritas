---
type: llm
---

PASS if the response covers both situations:
1. The application intentionally has no sensitive detectors: set
   `sd.enabled = false` in the Celeritas `SetupOptions` (or the equivalent
   `/celer/detector/enabled false` macro).
2. The application *does* have sensitive detectors: they are not visible
   to Celeritas on the master thread. They must be attached (e.g. with
   `SetSensitiveDetector` in `ConstructSDandField`) so that they exist when
   Celeritas sets up, or the logical volumes must be listed explicitly in
   `sd.force_volumes`.

FAIL if it only covers one of the two situations, or recommends something
incorrect such as disabling Celeritas entirely.
