.. Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _accel:

Acceleritas
===========

The ``accel`` directory contains components exclusive to coupling Celeritas
with Geant4 for user-oriented integration. A simple interface for multithreaded
or serial applications is demonstrated in :ref:`example_geant`, and the more
advanced implementation can be inspected in the :ref:`celer-g4` app.

.. _api_accel_high_level:

High-level interface
--------------------

The SimpleOffload class is an extremely easy-to-use interface for
offloading tracks to Celeritas in a multithreaded or serial application. The
class names correspond to user actions and ActionInitialization. It requires a
few app-owned pieces such as SharedParams and LocalTransporter to be owned by
the calling application; the options described below must also be set up and
provided.

.. doxygenclass:: celeritas::SimpleOffload

The SetupOptionsMessenger can be instantiated with a reference to a global
SetupOptions instance in order to provide a Geant4 "UI" macro interface to an
app's Celeritas options.

.. doxygenclass:: celeritas::SetupOptionsMessenger
   :members: none

Celeritas setup
---------------

The setup options help translate the Geant4 physics and problem setup to
Celeritas. They are also necessary to set up the GPU offloading
characteristics. Future versions of Celeritas will automate more of these
settings.

.. doxygenstruct:: celeritas::SetupOptions

.. doxygenstruct:: celeritas::SDSetupOptions

.. doxygenclass:: celeritas::UniformAlongStepFactory

.. doxygenclass:: celeritas::RZMapFieldAlongStepFactory

Detailed interface
------------------

These classes are usually integrated into UserActions. The ``SimpleOffload``
interface above hides the complexity of these classes, or for more complex
applications you can choose to use these classes directly instead of it.

.. doxygenclass:: celeritas::SharedParams

.. doxygenclass:: celeritas::LocalTransporter

Interface utilities
-------------------

.. doxygenfunction:: celeritas::MakeMTLogger

.. doxygenclass:: celeritas::ExceptionConverter

.. doxygenclass:: celeritas::AlongStepFactoryInterface


Classes usable by Geant4
------------------------

These utilities are based on Celeritas data structures and capabilities but are
written to be usable both by the ``celer-g4`` app and potential other users.

.. doxygenclass:: celeritas::GeantSimpleCalo

.. doxygenclass:: celeritas::HepMC3PrimaryGenerator

.. doxygenclass:: celeritas::RZMapMagneticField

