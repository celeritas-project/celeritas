.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _accel:

Acceleritas
===========

The ``accel`` directory contains components exclusive to coupling Celeritas
with Geant4 for user-oriented integration. See the
:file:`app/demo-geant-integration` for a complete example of how to use
Celeritas to offload EM tracks to GPU and dispatch hits back to Geant4
sensitive detectors.

Utilities
------------

.. doxygenfunction:: celeritas::MakeMTLogger

.. doxygenclass:: celeritas::ExceptionConverter

Setup
-----

.. doxygenstruct:: celeritas::SetupOptions

.. doxygenstruct:: celeritas::SDSetupOptions

.. doxygenclass:: celeritas::AlongStepFactoryInterface

.. doxygenclass:: celeritas::UniformAlongStepFactory

Transport interface
-------------------

These classes are usually integrated into UserActions.

.. doxygenclass:: celeritas::SharedParams

.. doxygenclass:: celeritas::LocalTransporter
