.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _inp_physics:

Physics
========

.. note:: For discussion of model and process naming, see issue 1562_ .

.. _1562: https://github.com/celeritas-project/celeritas/pull/1562

The selection and data used to create physics classes will be defined
underneath this struct.

.. celerstruct:: inp::Physics

Electromagnetic
---------------

.. celerstruct:: inp::EmPhysics

Optical
-------

Optical photon _generation_ is a part of the standard stepping loop that manages
EM, decay, and hadronic physics, but its _transport_ has its own separate
stepping loop, where surface physics is the most complex part. Therefore, the
``OpticalPhysics`` input includes optical photon _generation_ processes (such as
Cherenkov and scintillation) and surface physics information. The latter
describing how optical photons should interact with it.

.. celerstruct:: inp::OpticalPhysics

Celeritas' surface physics implementation is designed differently from Geant4
and is meant to reduce code complexity and improve extensibility.
In Geant4, a single model (e.g. "unified") is a hard-coded combination of:

- surface roughness types (e.g., polished or Gaussian),
- reflectivity calculation (whether to use a grid or not), and
- interaction description (e.g. dielectric-dielectric).

Celeritas separates these out into fully configurable surface physics inputs,
so that a given surface ID must be present in three separate surface models.
This "model unfolding" leads to a less simple input definition compared to
Geant4, but it allows for user-extensible surface physics models.

.. celerstruct:: inp::SurfacePhysics

Reflection
^^^^^^^^^^

.. celerstruct:: inp::GridReflection


Processes
---------

.. celerstruct:: inp::BremsstrahlungProcess
.. celerstruct:: inp::PairProductionProcess
.. celerstruct:: inp::PhotoelectricProcess
.. celerstruct:: inp::AtomicRelaxation


Models
------

.. celerstruct:: inp::SeltzerBergerModel
.. celerstruct:: inp::RelBremsModel
.. celerstruct:: inp::MuBremsModel
.. celerstruct:: inp::BetheHeitlerProductionModel
.. celerstruct:: inp::MuPairProductionModel
.. celerstruct:: inp::LivermorePhotoModel


.. _inp_grid:

Grids
-----

Tabulated physics data such as cross sections or energy loss are stored on
increasing, sorted 1D or 2D grids. Both linear and spline interpolation are supported.

.. celerstruct:: inp::Grid
.. celerstruct:: inp::UniformGrid
.. celerstruct:: inp::TwodGrid
.. celerstruct:: inp::Interpolation
