.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Detailed interface
------------------

These classes manage the low-level runtime interface between Celeritas and
Geant4.

.. doxygenclass:: celeritas::SharedParams
.. doxygenclass:: celeritas::LocalTransporter

Interface utilities
^^^^^^^^^^^^^^^^^^^

.. celerstruct:: AlongStepFactoryInput

.. doxygenclass:: celeritas::ExceptionConverter
.. doxygenclass:: celeritas::AlongStepFactoryInterface

.. _api_accel_adapters:

Classes usable by Geant4
^^^^^^^^^^^^^^^^^^^^^^^^

These utilities are based on Celeritas data structures and capabilities but are
written to be usable both by the ``celer-g4`` app and potential other users.

Fields
""""""

.. doxygenclass:: celeritas::RZMapMagneticField
.. doxygenclass:: celeritas::CylMapMagneticField
.. doxygenfunction:: celeritas::MakeCylMapFieldInput

Primary generators
""""""""""""""""""

.. doxygenclass:: celeritas::HepMC3PrimaryGenerator
.. doxygenclass:: celeritas::PGPrimaryGeneratorAction

Physics lists
"""""""""""""

Two physics constructors build exclusively processes supported by Celeritas for
Geant4:

.. doxygenclass:: celeritas::SupportedEmStandardPhysics
.. doxygenclass:: celeritas::SupportedOpticalPhysics

Two "modular" physics lists (one using Geant4 hadronics, the other using pure
Celeritas) are stand-ins for physics factories suitable for sending to
``G4RunManager::SetUserInitialization``.

.. doxygenclass:: celeritas::EmPhysicsList
.. doxygenclass:: celeritas::FtfpBertPhysicsList

.. _api_geant4_physics_options:

Physics setup
"""""""""""""

The input options incorporate process and model selection as well as default EM
parameters to send to Geant4.

.. celerstruct:: GeantPhysicsOptions

Sensitive detectors
"""""""""""""""""""

.. doxygenclass:: celeritas::GeantSimpleCalo
