.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _runtime:

Runtime
-------

ORANGE runtime tracking will be described here in greater detail in the future.

Surfaces
^^^^^^^^

.. highlight:: none

.. doxygenclass:: celeritas::ConeAligned
.. doxygenclass:: celeritas::CylAligned
.. doxygenclass:: celeritas::CylCentered
.. doxygenclass:: celeritas::GeneralQuadric
.. doxygenclass:: celeritas::Involute
.. doxygenclass:: celeritas::Plane
.. doxygenclass:: celeritas::PlaneAligned
.. doxygenclass:: celeritas::SimpleQuadric
.. doxygenclass:: celeritas::Sphere
.. doxygenclass:: celeritas::SphereCentered

.. highlight:: cpp

Acceleration structures
^^^^^^^^^^^^^^^^^^^^^^^

Celeritas uses a bounding interval hierarchy to accelerate volume
intersections.

Navigation interface
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::OrangeTrackView
