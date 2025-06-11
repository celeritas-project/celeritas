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
are an instantiation of the volume *below* inside the volume *above*. This
instantiation is associated with a transformation and
other metadata. In HEP geometries, child nodes may not overlap each other or
their enclosing parent volume.

.. table:: Celeritas nomenclature tends toward computer science terminology.

   +------------------+-----------+---------------------------------------------+
   | Celeritas        | VecGeom   | Geant4                                      |
   +==================+===========+=============================================+
   | Volume           | Unplaced  | Logical volume                              |
   +------------------+-----------+---------------------------------------------+
   | Volume instance  | Placed    | Physical volume (plus copy number)          |
   +------------------+-----------+---------------------------------------------+
   | Child            | Daughter  | Daughter                                    |
   +------------------+-----------+---------------------------------------------+
   | Parent           | Mother    | Mother                                      |
   +------------------+-----------+---------------------------------------------+


Celeritas defines abstract geometry concepts, indexed as IDs, to support
multiple geometry applications [#]_ and to make the code backend-agnostic for
integrating with physics. These include "volumes" (known in some other
fields as "cells").

.. [#] In the future the use of these abstract concepts will enable detector
   descriptions, and geometry models for other applications, that are *not*
   Geant4 hierarchies.

Volume
   A *volume* corresponds to a homogeneous physical object that can have multiple
   instances but is treated identically. It has a specific shape, material,
   metadata, and associated scoring/sensitive region. Each volume is
   simply a *node* in the detector geometry graph. This definition differs
   slightly from Geant4 and VecGeom, where the ``G4LogicalVolume`` and
   ``UnplacedVolume`` classes directly reference the child geometry nodes and
   thus implicitly include the objects embedded in a volume.

Volume instance
   An *instance* of a volume is defined in conjunction with a transform and an
   enclosing object (or, in the special case of the outermost or "world" volume
   instance, no enclosing object). In Geant4 this roughly corresponds to a
   physical volume. [#]_ VecGeom refers to volume instances as *placed
   volumes*. In ORANGE for KENO, this would correspond to a hole
   placement, array element, or local media. The volume instance is an *edge* in
   the graph of volumes.

Unique instance
   A *unique* instance of a volume refers to the logical definition of a
   specific region of global space in the geometry model. It is the full
   directed path :cite:`bender-listsdecisions-2010` from the root volume node
   (world volume) to a node (logical volume) somewhere in the graph, thereby
   describing all enclosing volumes and their locations. This path can be
   encoded uniquely as a single integer by pre-calculating the number of direct
   and indirect children for each node.  Celeritas always uses 64-bit integers
   to store the ``VolumeUniqueInstanceId``.

.. [#] A ``VolumeInstanceId`` has a one-to-one mapping for ``G4PVPlacement``,
   but "replica" and "parameterized" volumes use a single physical volume to
   represent multiple spatial elements. For those, we currently define a
   :cpp:struct:`celeritas::GeantPhysicalInstance` that is a tuple of
   physical volume and a replica instance. Eventually that will become an
   implementation detail.


.. toctree::
   :maxdepth: 2

   geometry/interfaces.rst
   geometry/geant4.rst
   geometry/orange.rst
   geometry/vecgeom.rst
