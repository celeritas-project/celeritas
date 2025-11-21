.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _runtime:

Runtime
-------

ORANGE runtime execution will be described here in greater detail in the future.

Acceleration structures
^^^^^^^^^^^^^^^^^^^^^^^

Celeritas uses a bounding interval hierarchy to accelerate volume
intersections.

Tracking
^^^^^^^^

There are three slightly different algorithms in ORANGE for finding the
distance to the next point on a surface on an outer boundary of the volume.

1. The "simple" algorithm is for volumes that are the intersection of a number
   of infinite surfaces. All convex volumes are simple, but because quadric
   surfaces are allowed all simple surfaces do not have to be convex.
   Intersecting any surface by definition causes the volume to be exited. (A
   single "blade" of the ATLAS EMEC accordion, a "twisted trapezoid", has
   hyperbolic paraboloids on its axial faces. It is simple but not
   convex. The same is true for a "ring" formed by subtracting two cylinders of
   the same height.)
2. The "internal surfaces" algorithm has to track through and discard any
   surfaces that do not result in the CSG volume changing. Crossing an internal
   surface is changing from one node inside the CSG tree to another.
3. Each universe may have a single "background" volume that is selected if a
   point is not inside any interior volume.

Interface
^^^^^^^^^

ORANGE runtime routines are always invoked via the track view, which
encapsulates geometry navigation for a single particle track.

.. doxygenclass:: celeritas::OrangeTrackView
