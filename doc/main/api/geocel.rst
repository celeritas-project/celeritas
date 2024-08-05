.. Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_geocel:

Geometry classes
================

These classes, and the analogous :ref:`api_orange` class, provide a unified
interface to model properties needed to set up the problem and print output.

Geometry interfaces
-------------------

.. doxygenclass:: celeritas::GeoParamsInterface
   :members:

.. doxygenclass:: celeritas::VecgeomParams

.. doxygenclass:: celeritas::GeantGeoParams

Geant4 geometry utilities
-------------------------

.. doxygenfunction:: celeritas::load_geant_geometry
.. doxygenfunction:: celeritas::find_geant_volumes

.. doxygenclass:: celeritas::g4vg::Converter

