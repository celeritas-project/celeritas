.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

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

.. celerstruct:: AlongStepFactoryInput

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

.. _api_geant4_physics_options:

Physics constructors
^^^^^^^^^^^^^^^^^^^^

A Geant4 physics constructor :cpp:class:`celeritas::SupportedEmStandardPhysics` allows
very fine-grained selection of the EM physics processes supported by Celeritas.
The input options incorporate process and model selection as well as default EM
parameters to send to Geant4.

.. celerstruct:: GeantPhysicsOptions
.. doxygenclass:: celeritas::SupportedEmStandardPhysics

Physics lists
^^^^^^^^^^^^^

Two physics lists (one using Geant4 hadronics, the other using pure Celeritas)
allow setup of EM physics using only processes supported by Celeritas.

.. doxygenclass:: celeritas::EmPhysicsList
.. doxygenclass:: celeritas::FtfpBertPhysicsList

Sensitive detectors
^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::GeantSimpleCalo
