.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Low-level Celeritas integration
===============================

This subsection contains details of importing Geant4 data into Celeritas.

Geant4 geometry utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::GeantGdmlLoader
.. doxygenfunction:: celeritas::load_gdml
.. doxygenfunction:: celeritas::save_gdml
.. doxygenfunction:: celeritas::find_geant_volumes

Geant4 physics interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::GeantImporter

.. doxygenclass:: celeritas::GeantSetup

.. _api_geant4_physics_options:

Geant4 physics options
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: celeritas::GeantPhysicsOptions
   :members:
   :no-link:


