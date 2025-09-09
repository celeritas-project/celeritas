.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Geant4 geometry
===============

Celeritas defines mappings to the underlying Geant4 geometry objects for
integration in the rest of the code base. The
:cpp:class:`celeritas::GeantGeoParams` class manages these mappings.
Additionally, like the VecGeom and ORANGE geometry engines, it supports
navigation for individual track states (though only on CPU, and without full
support for field navigation).

.. doxygenclass:: celeritas::GeantGeoParams

Runtime interface
-----------------

.. doxygenclass:: celeritas::GeantGeoTrackView
