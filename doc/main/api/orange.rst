.. Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_orange:

ORANGE
======

The ORANGE (Oak Ridge Advanced Nested Geometry Engine) package is currently
under development as the version in SCALE is ported to GPU.

Runtime interfaces
------------------

.. doxygenclass:: celeritas::OrangeParams

.. doxygenclass:: celeritas::OrangeTrackView


Geometry creation
-----------------

ORANGE geometry can (TODO: not yet, but eventually) be constructed from
multiple geometry representations, including Geant4 HEP geometry.

CSG unit
^^^^^^^^

The CSG *unit* is a general scene comprising arbitrary volumes made of arbitrary
quadric faces. The name "unit" is derived from the KENO criticality safety
code, where a unit is a reusable composable building block for arrays.

Each unit is constructed from the user defining ``ObjectInterface``
implementations and relationships, and specifying which of them are volumes.
The Object interface is implemented by:

Shape
   A convex, finite region of space defined by the intersection of multiple
   quadric surfaces. The Shape is implemented using a single ConvexRegion,
   which is an implementation that builds the underlying surfaces and bounding
   boxes. Shapes should be as simple as possible, aligned along and centered on
   the Z axis.
Solid
   A shape that's hollowed out and/or has a slice removed. It is equivalent to
   a CSG operation on two shapes of the same type and an azimuthal wedge.
ExtrudedSolid
   NOT YET IMPLEMENTED: a union of transformed solids along the Z axis, which
   can also be hollowed and sliced.
Transformed
   Applies a transform to another CSG object.
AnyObjects, AllObjects, and NegatedObject
   Apply the CSG operations of union, intersection, and negation. The first two
   are implemented as templates of a JoinObjects class.

Objects are typically constructed and used as shared pointers so that they can
be reused in multiple locations.

.. doxygenclass:: celeritas::orangeinp::Shape

.. doxygenclass:: celeritas::orangeinp::Solid

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
       #ConvexRegion const& interior()*
     }
     <<Abstract>> ShapeBase

     class Shape {
       -string label;
       -ConvexRegion region;
     }
     Shape *-- ConvexRegion

     class ConvexRegion {
       +void build(ConvexSurfaceBuilder&)*
     }
     <<Interface>> ConvexRegion
     ConvexRegion <|-- Box
     ConvexRegion <|-- Sphere

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

CSG unit construction
^^^^^^^^^^^^^^^^^^^^^

The Object classes above are all factory functions for creating a CSG tree and
transformed surfaces corresponding to leaf nodes. Some important aspects of
this construction process are:

- Transforming constructed surfaces based on the stack of transformations
- Simplifying and normalizing surfaces (e.g., ensuring planes are pointing in a
  "positive" direction and converting arbitrary planes to axis-aligned planes)
- Deduplicating "close" surfaces to eliminate boundary crossing errors
- Naming constructed surfaces based on the constructing surface type
- Constructing bounding boxes using the original and simplified surfaces, as
  well as additional specifications from the convex regions
- Adding surfaces as leaf nodes to the CSG tree, and defining additional nodes
  based on those
- Simplifying the CSG tree based on boundary conditions and other factors

Geant4 geometry translation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Geant4 geometry is a hierarchy of "logical volumes" comprised of solids.
Deeper ("daughter") volumes are "placed" into a parent ("mother") volume after
applying a transformation (translation, rotation, reflection, or a
combination), displacing the material in the parent volume. Besides this
displacement, no overlap is allowed.

Solids are parametrized volumes that may be hollowed out, have slices removed,
or be defined as a CSG operation on placed volumes. They are sometimes but not
always convex. See the `Geant4 documentation`_ for descriptions of all the
predefined solids.

A logical volume can be referenced multiple times, i.e., placed multiple times in
multiple different volumes. The Geant4-ORANGE converter decomposes (TODO: not
yet implemented) the graph of logical volume relationships into subgraphs that
each become a CSG unit. The decomposition should minimize the number of
subgraphs while minimizing (eliminating even?) the number of interior nodes
with multiple incoming edges, i.e., the number of solids that have to be
duplicated *within* a unit.

.. _Geant4 documentation: https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/index.html

