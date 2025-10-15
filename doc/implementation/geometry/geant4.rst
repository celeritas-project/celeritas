.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_geant4_geo:

Geant4 geometry
===============

Celeritas defines mappings to the underlying Geant4 geometry objects for
integration in the rest of the code base. The
:cpp:class:`celeritas::GeantGeoParams` class manages these mappings.
Additionally, like the VecGeom and ORANGE geometry engines, it supports
navigation for individual track states (though only on CPU, and without full
support for field navigation).

.. doxygenclass:: celeritas::GeantGeoParams

Integration with Celeritas
--------------------------

.. table:: Celeritas geometry IDs and their analogous Geant4 classes.

   =================== =========================
   Celeritas           Geant4
   =================== =========================
   VolumeInstanceId    G4VPhysicalVolume
   VolumeId            G4LogicalVolume
   GeoMatId            G4Material
   SurfaceId           G4LogicalSurface
   =================== =========================


Runtime interface
-----------------

.. doxygenclass:: celeritas::GeantGeoTrackView
