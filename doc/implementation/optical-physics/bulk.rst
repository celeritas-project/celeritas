.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _optical_bulk:

Bulk physics
============

Volumetric optical photon processes are commonly known as "bulk" physics as
they happen in the bulk of the material. These processes are stochastic and
based on the user-provided, wavelength-dependent mean free paths in the optical
materials.

.. doxygenclass:: celeritas::optical::PhysicsParams

Built-in processes
------------------

The ``Interactor`` classes encapsulate the physics models for each physical
process.

.. doxygenclass:: celeritas::optical::AbsorptionInteractor
.. doxygenclass:: celeritas::optical::RayleighInteractor
.. doxygenclass:: celeritas::optical::MieInteractor
.. doxygenclass:: celeritas::optical::WavelengthShiftInteractor

Mean free paths
---------------

.. doxygenclass:: celeritas::optical::RayleighMfpCalculator
