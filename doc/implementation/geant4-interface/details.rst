.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Detailed interface
------------------

These classes are usually integrated into UserActions. The ``SimpleOffload``
interface above hides the complexity of these classes, or for more complex
applications you can choose to use these classes directly instead of it.

.. doxygenclass:: celeritas::SharedParams
   :members:
   :no-link:

.. doxygenclass:: celeritas::LocalTransporter
   :members:
   :no-link:

Interface utilities
-------------------

.. doxygenfunction:: celeritas::MakeMTLogger

.. doxygenclass:: celeritas::ExceptionConverter

.. doxygenstruct:: celeritas::AlongStepFactoryInput

.. doxygenclass:: celeritas::AlongStepFactoryInterface


.. _api_accel_adapters:

Classes usable by Geant4
------------------------

These utilities are based on Celeritas data structures and capabilities but are
written to be usable both by the ``celer-g4`` app and potential other users.

Fields
^^^^^^

.. doxygenclass:: celeritas::RZMapMagneticField

.. doxygenclass:: celeritas::CylMapMagneticField

.. doxygenfunction:: celeritas::MakeCylMapFieldInput

Primary generators
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::HepMC3PrimaryGenerator

.. doxygenclass:: celeritas::PGPrimaryGeneratorAction

Physics lists
^^^^^^^^^^^^^

Two physics lists (one using Geant4 hadronics, the other using pure Celeritas)
allow setup of

.. doxygenstruct:: celeritas::GeantPhysicsOptions
   :members:
   :no-link:

.. doxygenclass:: celeritas::EmPhysicsList

.. doxygenclass:: celeritas::FtfpBertPhysicsList

Sensitive detectors
^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::GeantSimpleCalo
