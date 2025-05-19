.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_geometry:

********
Geometry
********

Detector geometry descriptions for HEP are almost universally defined using a
hierarchy of fully nested volumes, often saved as a GDML file
:cite:`gdml-2006`. These volumes can be represented as a directional
acyclic graph (DAG): the nodes are the geometric elements, and the edges
are a geometry placement (which is associated with a transformation and
other metadata). In HEP geometries, child nodes may not overlap each other or
their enclosing parent [#]_ volume.

.. [#] The Geant4 terms "daughter" and "mother" correspond to "child volume"
   and "parent volume" in Celeritas.

To support multiple geometry applications (including detector descriptions that
are not Geant4 hierarchies), and to make the code
backend-agnostic for integrating with physics, Celeritas defines abstract
geometry concepts indexed as IDs.

VolumeId
  A volume corresponds to a physical object that can have multiple instances
  but is treated (almost always) identically: it has the same shape, material,
  and associated scoring/sensitive region. In Geant4 this is a "logical
  volume", and VecGeom refers to these as "unplaced volumes". This is simply a
  *node* in the graph of volumes.

VolumeInstanceId
  A volume *instance* is a unique placement of a volume in the geometry
  hierarchy. The instance usually has a transformation to the enclosing
  volume's coordinate system. In Geant4 this roughly corresponds to a physical
  volume. (It corresponds exactly to a ``G4PVPlacement``, but "replica" and
  "parameterized" volumes use a single physical volume to represent multiple
  spatial elements. For those, we define a
  :cpp:struct:`celeritas::GeantPhysicalInstance` that is a tuple containing a
  physical volume and a replica instance.) VecGeom refers to volume instances
  as *placed volumes*. In ORANGE for KENO, this would correspond to a hole
  placement, array element, or local media. The volume instance is an *edge* in
  the graph of volumes.

VolumeUniqueInstanceId
  A "unique" volume instance refers to a unique region of global space in the
  geometry model. It is the full *directed trail* from the root volume node to
  a node somewhere in the graph, thereby describing all enclosing volumes and
  their locations. This path can be encoded uniquely as a single integer
  by pre-calculating the number of direct and indirect children for each node.
  Celeritas always uses 64-bit integers to store this unique instance ID.


.. doxygenstruct:: celeritas::GeantPhysicalInstance

.. doxygenclass:: celeritas::GeoParamsInterface


.. toctree::
   :maxdepth: 2

   geometry/geant4.rst
   geometry/orange.rst
   geometry/vecgeom.rst
