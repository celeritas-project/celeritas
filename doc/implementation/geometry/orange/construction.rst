.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Construction
------------

ORANGE input can be manually constructed via an API for testing and direct
integration with other applications.

Fundamental shapes
^^^^^^^^^^^^^^^^^^

The lowest level primitive for construction is the "intersect region," which is
a CSG intersection of half-spaces. The IntersectRegion interface
helps construct these objects with a ``build`` method that constructs surfaces
and a CSG intersection node.

.. doxygenclass:: celeritas::orangeinp::IntersectRegionInterface

.. doxygenclass:: celeritas::orangeinp::Box
.. doxygenclass:: celeritas::orangeinp::Cone
.. doxygenclass:: celeritas::orangeinp::CutCylinder
.. doxygenclass:: celeritas::orangeinp::Cylinder
.. doxygenclass:: celeritas::orangeinp::Ellipsoid
.. doxygenclass:: celeritas::orangeinp::EllipticalCylinder
.. doxygenclass:: celeritas::orangeinp::EllipticalCone
.. doxygenclass:: celeritas::orangeinp::ExtrudedPolygon
.. doxygenclass:: celeritas::orangeinp::GenPrism
.. doxygenclass:: celeritas::orangeinp::Hyperboloid
.. doxygenclass:: celeritas::orangeinp::InfPlane
.. doxygenclass:: celeritas::orangeinp::InfAziWedge
.. doxygenclass:: celeritas::orangeinp::InfPolarWedge
.. doxygenclass:: celeritas::orangeinp::Involute
.. doxygenclass:: celeritas::orangeinp::Paraboloid
.. doxygenclass:: celeritas::orangeinp::Parallelepiped
.. doxygenclass:: celeritas::orangeinp::Prism
.. doxygenclass:: celeritas::orangeinp::Sphere

.. _api_orange_objects:

Objects
^^^^^^^

Each unit is constructed from the user defining ``ObjectInterface``
implementations and relationships, and specifying which of them are volumes.
The Object interface is implemented by:

Shape
   A finite (and usually convex) region of space defined by the intersection of
   multiple quadric surfaces. The Shape is implemented using a single
   IntersectRegion,
   which is an implementation that builds the underlying surfaces and bounding
   boxes. Shapes should be as simple as possible, aligned along and
   usually centered on the *z* axis.
Solid
   A shape that's hollowed out and/or has a slice removed. It is equivalent to
   a CSG operation on two shapes of the same type and an azimuthal wedge.
PolySolid
   A union of transformed solids along the *z* axis, which can also be hollowed
   and sliced azimuthally.
RevolvedPolygon
   A convex or concave polygon, revolved around the *z* axis, which can be sliced azimuthally.
StackedExtrudedPolygon
   A convex or concave polygon, extruded along a polyline, with scaling applied
   at each polyline point.
Transformed
   Applies a transformation (rotation, translation) to another CSG object.
AnyObjects, AllObjects, and NegatedObject
   Apply the CSG operations of union, intersection, and negation. The first two
   are implemented as templates of a JoinObjects class.

Objects are typically constructed and used as shared pointers so that they can
be reused in multiple locations. :numref:`fig-orangeinp-types` summarizes these types.

.. _fig-orangeinp-types:

.. figure:: /_static/dot/orangeinp-types.*
   :align: center
   :width: 80%

   Examples of objects and "intersect regions" used by ORANGE input
   preprocessing.


.. highlight:: cpp

.. doxygenclass:: celeritas::orangeinp::EnclosedAzi
.. doxygenclass:: celeritas::orangeinp::EnclosedPolar
.. doxygenclass:: celeritas::orangeinp::Shape
.. doxygenclass:: celeritas::orangeinp::Solid
.. doxygenclass:: celeritas::orangeinp::Truncated

.. highlight:: none

.. doxygenclass:: celeritas::orangeinp::PolySegments
.. doxygenclass:: celeritas::orangeinp::PolyCone
.. doxygenclass:: celeritas::orangeinp::PolyPrism
.. doxygenclass:: celeritas::orangeinp::RevolvedPolygon
.. doxygenclass:: celeritas::orangeinp::StackedExtrudedPolygon

.. highlight:: cpp

.. doxygenclass:: celeritas::orangeinp::Transformed

.. doxygenclass:: celeritas::orangeinp::NegatedObject
.. doxygenclass:: celeritas::orangeinp::JoinObjects

.. doxygenfunction:: celeritas::orangeinp::make_subtraction
.. doxygenfunction:: celeritas::orangeinp::make_rdv

.. _orange_csg_unit:

CSG unit
^^^^^^^^

The CSG *unit* is a general scene comprising arbitrary volumes made of arbitrary
quadric and planar faces. The name "unit" is derived from the KENO criticality
safety code :cite:`kenovi`, where a unit is a reusable composable building
block for arrays.

The Object classes above are all factory functions for creating a CSG tree and
transformed surfaces corresponding to leaf nodes. Some important aspects of
this construction process are:

- Transforming constructed surfaces based on the stack of transformations
- Simplifying and normalizing surfaces (e.g., ensuring planes are pointing in a
  "positive" direction and converting arbitrary planes to axis-aligned planes)
- De-duplicating "close" surfaces to eliminate boundary crossing errors
- Naming constructed surfaces based on the constructing surface type
- Constructing bounding boxes using the original and simplified surfaces, as
  well as additional specifications from the convex regions
- Adding surfaces as leaf nodes to the CSG tree, and defining additional nodes
  based on those
- Simplifying the CSG tree based on boundary conditions and other factors

The *proto* type, short for "proto-universe," has a one-to-one correspondence
to a universe and constructs its input with a ``build`` function.

.. doxygenclass:: celeritas::orangeinp::UnitProto
