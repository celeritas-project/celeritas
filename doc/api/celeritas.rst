.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

Celeritas
=========

The ``celeritas`` directory focuses on the physics and transport loop
implementation for the Celeritas codebase, using components from the
``corecel`` and ``orange`` dependencies.

Problem definition
------------------

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams

Transport interface
-------------------

.. doxygenclass:: celeritas::Stepper

External code interfaces
------------------------

.. doxygenclass:: celeritas::GeantImporter

.. doxygenclass:: celeritas::GeantSetup

.. doxygenclass:: celeritas::VecgeomParams

On-device access
----------------

.. doxygenclass:: celeritas::MaterialTrackView

.. doxygenclass:: celeritas::ParticleTrackView

.. doxygenclass:: celeritas::PhysicsTrackView

