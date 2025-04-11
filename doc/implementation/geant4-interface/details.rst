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

.. doxygenclass:: celeritas::GeantSimpleCalo

.. doxygenclass:: celeritas::HepMC3PrimaryGenerator

.. doxygenclass:: celeritas::RZMapMagneticField

.. doxygenclass:: celeritas::CylMapMagneticField

.. doxygenfunction:: celeritas::MakeCylMapFieldInput

