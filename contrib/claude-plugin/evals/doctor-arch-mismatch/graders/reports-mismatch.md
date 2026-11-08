---
type: llm
---

The installation reports `gpu_architectures` "70" while the device has
compute capability 80.

PASS if the response states that the GPU's compute capability (80, i.e.
sm_80) is not among the architectures Celeritas was compiled for (70), that
this can make kernels fail at run time, and that Celeritas (and VecGeom)
should be rebuilt for architecture 80.

FAIL if it does not notice the mismatch or declares the installation fine.
