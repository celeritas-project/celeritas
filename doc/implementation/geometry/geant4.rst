.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_geant4_geo:

Geant4 geometry
===============

The Geant4 geometry is a hierarchy of "logical volumes" comprised of solids.
Child ("daughter") volumes are "placed" into a parent ("mother") volume after
applying a transformation (translation, rotation, reflection, or a
combination), displacing the material in the parent volume. Besides this
displacement, no overlap is allowed.

Solids are parametrized volumes that may be hollowed out, have slices removed,
or be defined as a CSG operation on placed volumes. They are sometimes but not
always convex. See the `Geant4 documentation`_ for descriptions of all the
predefined solids.

.. _Geant4 documentation: https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/index.html

Integration with Celeritas
--------------------------

Celeritas defines GPU-friendly mappings (see :numref:`tab-geo-ids`) to the
underlying Geant4 geometry objects for integration in the rest of the code
base.
The :cpp:class:`celeritas::GeantGeoParams` class manages these mappings.
Additionally, like the VecGeom and ORANGE geometry engines, it supports
navigation for individual track states (though only on CPU, and without full
support for field navigation).

.. doxygenclass:: celeritas::GeantGeoParams

.. _tab-geo-ids:

.. table:: Celeritas geometry IDs and their analogous Geant4 classes.

   ====================== ============================
   Celeritas              Geant4
   ====================== ============================
   VolumeUniqueInstanceId G4TouchableHistory
   VolumeInstanceId       G4VPhysicalVolume
   VolumeId               G4LogicalVolume
   GeoMatId               G4Material
   SurfaceId              G4LogicalSurface
   ====================== ============================


Runtime interface
-----------------

.. doxygenclass:: celeritas::GeantGeoTrackView
