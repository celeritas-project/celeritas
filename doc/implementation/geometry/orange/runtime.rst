.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _runtime:

Runtime
-------

ORANGE runtime execution will be described here in greater detail in the future.

Acceleration structures
^^^^^^^^^^^^^^^^^^^^^^^

Celeritas uses a bounding volume hierarchy to accelerate volume
intersections. Bounding box intersections in the hierarchy use a highly
optimized testing function.

.. doxygenfunction:: celeritas::intersects_segment

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

Initialization
^^^^^^^^^^^^^^

A track is initialized by searching (using the acceleration structure) for the
volume whose CSG logic is satisfied by the senses of its faces at the
starting position. A point exactly on one of those surfaces has no sense of its
own, so during initialization only, the sense is taken to be the side that the
track is moving toward: *outside* if :math:`\hat{n} \cdot \hat{\Omega} > 0`
and *inside* if :math:`\hat{n} \cdot \hat{\Omega} < 0`, where
:math:`\hat{n}` is the outward normal of the surface at that point and
:math:`\hat{\Omega}` is the track direction. The track is then logically on
that surface (``boundary_out``), exactly as if it had just crossed it, so the
next intersection search ignores that surface. A track starting exactly on a
surface and moving exactly tangent to it cannot be placed on either side, and
initialization fails.

Interface
^^^^^^^^^

ORANGE runtime routines are always invoked via the track view, which
encapsulates geometry navigation for a single particle track.

.. doxygenclass:: celeritas::OrangeTrackView
