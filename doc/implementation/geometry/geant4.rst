.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Geant4 geometry
===============

Celeritas defines mappings to the underlying Geant4 geometry objects for
integration in the rest of the code base. There is limited support for
executing the main simulation engine using Geant4 navigation states, but this
class is to be used as an interface between the Celeritas indexing and an
in-memory Geant4 geometry.

.. doxygenclass:: celeritas::GeantGeoParams
