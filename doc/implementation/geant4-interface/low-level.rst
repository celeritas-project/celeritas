.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Low-level Celeritas integration
===============================

This subsection contains details of importing Geant4 data into Celeritas.

Geant4 geometry utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

These utility classes are used to set up the Geant4 global geometry state.

.. doxygenclass:: celeritas::GeantGdmlLoader
.. doxygenfunction:: celeritas::load_gdml
.. doxygenfunction:: celeritas::save_gdml
.. doxygenfunction:: celeritas::find_geant_volumes

Geant4 physics interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::GeantImporter
.. doxygenclass:: celeritas::GeantSetup
