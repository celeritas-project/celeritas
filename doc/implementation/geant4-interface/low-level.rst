.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Low-level integration utilities
-------------------------------

This subsection contains details of integrating Geant4 with Celeritas.

Geometry utilities
^^^^^^^^^^^^^^^^^^

These utility classes are used to set up the Geant4 global geometry state.

.. doxygenclass:: celeritas::GeantGdmlLoader
.. doxygenfunction:: celeritas::load_gdml
.. doxygenfunction:: celeritas::save_gdml
.. doxygenfunction:: celeritas::find_geant_volumes

Physics interfaces
^^^^^^^^^^^^^^^^^^

This will be replaced by other utilities in conjunction with the
:ref:`problem input <input>`.

.. doxygenclass:: celeritas::GeantImporter
.. doxygenclass:: celeritas::GeantSetup


.. _g4_utils:

Utility interfaces
^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: celeritas::MakeMTSelfLogger
.. doxygenfunction:: celeritas::MakeMTWorldLogger

.. doxygenclass:: celeritas::ScopedGeantLogger
.. doxygenclass:: celeritas::ScopedGeantExceptionHandler
