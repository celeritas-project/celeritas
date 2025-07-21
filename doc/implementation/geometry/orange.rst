.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_orange:

ORANGE
======

The Oak Ridge Advanced Nested Geometry Engine (ORANGE)
:cite:`orange-tm` is a surface-based geometry that has been adapted to GPU
execution to support platform portability in Celeritas. It can be built via its
interface to SCALE or constructed automatically from Geant4 geometry
representation.

Construction API
----------------

ORANGE input can be manually constructed via an API for testing and direct
integration with other applications.

Intersect region
^^^^^^^^^^^^^^^^

The lowest level primitive for construction is the "intersect region," which is
a CSG intersection of half-spaces. The IntersectRegion interface
helps construct these objects.

.. doxygenclass:: celeritas::orangeinp::IntersectRegionInterface
.. doxygenclass:: celeritas::orangeinp::Box
.. doxygenclass:: celeritas::orangeinp::Cone
.. doxygenclass:: celeritas::orangeinp::Cylinder
.. doxygenclass:: celeritas::orangeinp::Ellipsoid
.. doxygenclass:: celeritas::orangeinp::EllipticalCylinder
.. doxygenclass:: celeritas::orangeinp::EllipticalCone
.. doxygenclass:: celeritas::orangeinp::ExtrudedPolygon
.. doxygenclass:: celeritas::orangeinp::GenPrism
.. doxygenclass:: celeritas::orangeinp::InfPlane
.. doxygenclass:: celeritas::orangeinp::InfAziWedge
.. doxygenclass:: celeritas::orangeinp::Involute
.. doxygenclass:: celeritas::orangeinp::Parallelepiped
.. doxygenclass:: celeritas::orangeinp::Prism
.. doxygenclass:: celeritas::orangeinp::Sphere

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
StackedExtrudedPolygon
   A convex or concave polygon, extruded along a polyline, with scaling applied
   at each polyline point.
Transformed
   Applies a transformation (rotation, translation) to another CSG object.
AnyObjects, AllObjects, and NegatedObject
   Apply the CSG operations of union, intersection, and negation. The first two
   are implemented as templates of a JoinObjects class.

Objects are typically constructed and used as shared pointers so that they can
be reused in multiple locations.

.. doxygenclass:: celeritas::orangeinp::Shape
.. doxygenclass:: celeritas::orangeinp::Solid
.. doxygenclass:: celeritas::orangeinp::Truncated

.. doxygenclass:: celeritas::orangeinp::PolyCone
.. doxygenclass:: celeritas::orangeinp::PolyPrism
.. doxygenclass:: celeritas::orangeinp::StackedExtrudedPolygon

.. doxygenclass:: celeritas::orangeinp::Transformed

.. doxygenclass:: celeritas::orangeinp::NegatedObject
.. doxygenclass:: celeritas::orangeinp::JoinObjects

.. doxygenfunction:: celeritas::orangeinp::make_subtraction
.. doxygenfunction:: celeritas::orangeinp::make_rdv


.. mermaid::

   classDiagram
     Object <|-- Transformed
     Object <|-- Shape
     Object <|-- NegatedObject
     Object <|-- JoinObjects
     ShapeBase <|-- Shape
     class Object {
       +string_view label()*
       +NodeId build(VolumeBuilder&)*
     }
     <<Interface>> Object
     class Transformed {
       -SPConstObject obj
       -VariantTransform transform
     }
     Transformed *-- Object

     class ShapeBase {
       #IntersectRegion const& interior()*
     }
     <<Abstract>> ShapeBase

     class Shape {
       -string label;
       -IntersectRegion region;
     }
     Shape *-- IntersectRegion

     class IntersectRegion {
       +void build(IntersectSurfaceBuilder&)*
     }
     <<Interface>> IntersectRegion
     IntersectRegion <|-- Box
     IntersectRegion <|-- Sphere

     class Box {
       -Real3 halfwidths
     }
     class Sphere {
       -real_type radius
     }

     Shape <|.. BoxShape
     Shape <|.. SphereShape

     BoxShape *-- Box
     SphereShape *-- Sphere

.. stop weird vim formatting here... |--|

CSG unit
^^^^^^^^

The CSG *unit* is a general scene comprising arbitrary volumes made of arbitrary
quadric and planar faces. The name "unit" is derived from the KENO criticality
safety code :cite:`kenovi`, where a unit is a reusable composable building
block for arrays.

.. doxygenclass:: celeritas::orangeinp::UnitProto


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

Geant4 geometry translation
---------------------------

The Geant4 geometry is a hierarchy of "logical volumes" comprised of solids.
Child ("daughter") volumes are "placed" into a parent ("mother") volume after
applying a transformation (translation, rotation, reflection, or a
combination), displacing the material in the parent volume. Besides this
displacement, no overlap is allowed.

Solids are parametrized volumes that may be hollowed out, have slices removed,
or be defined as a CSG operation on placed volumes. They are sometimes but not
always convex. See the `Geant4 documentation`_ for descriptions of all the
predefined solids.

A logical volume can be referenced multiple times, i.e., placed multiple times in
multiple different volumes. The Geant4-ORANGE converter decomposes the graph of
logical volume relationships into subgraphs that
each become a CSG unit. This decomposition is currently tuned so that:

- Volumes with no children are directly placed as "material" leaf nodes into a
  unit
- Logical volumes placed in a singular location without transforms are also
  placed as materials with child volumes explicitly subtracted out
- Union or poly volumes (for now!) must be placed as materials even if they are
  used multiple times and have daughter volumes.

.. _Geant4 documentation: https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/index.html

Runtime interfaces
------------------

.. doxygenclass:: celeritas::OrangeParams

.. doxygenclass:: celeritas::OrangeTrackView
