.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0


.. _api_physics_def:

Physics definition
==================

Celeritas contains several high-level "parameter" classes that allow setup-time
access to problem data. These classes all correspond directly to "TrackView"
classes (see the `developer documentation`_ for details).

.. _developer documentation: https://celeritas-project.github.io/celeritas/dev/classes.html

Celeritas parameters
--------------------

.. doxygenclass:: celeritas::MaterialParams
.. doxygenclass:: celeritas::ParticleParams
.. doxygenclass:: celeritas::PhysicsParams
.. doxygenclass:: celeritas::CutoffParams

Geant4 integration
------------------

The "physics" material ID in Celeritas is specified as a combination of the
actual model-specified material, which is the "geometry" material ID (see
:ref:`api_geometry`), and the region.

.. _table-physics-ids:

.. table:: Celeritas physics IDs their analogous Geant4 classes.

   ============ =========================
   Celeritas    Geant4
   ============ =========================
   ParticleId   G4ParticleDefinition
   PhysMatId    G4MaterialCutsCouple
   OptMatId     G4MaterialPropertiesTable
   PhysSurfId   G4OpticalSurface
   ProcessId    G4VProcess
   ============ =========================


.. _api_problem_setup:

Setting up problems
-------------------

Problem data is specified from applications and the Geant4 user interface using
the :ref:`input` API and loaded through "importers". Currently, Celeritas
relies on Geant4 and external Geant4 data files for its setup.

.. doxygennamespace:: celeritas::setup
