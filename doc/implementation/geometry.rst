.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_geometry:

********
Geometry
********

Detector geometry descriptions for HEP are almost universally defined using a
hierarchy of fully nested volumes, often saved as a GDML file
:cite:`gdml-2006`. These user-defined volumes can be represented as a directed
acyclic graph (DAG): the nodes are the geometric elements, and the edges
are an instantiation of the volume *below* inside the volume *above*. This
instantiation is associated with a transformation and
other metadata. In HEP geometries, child nodes may not overlap each other or
their enclosing parent volume.

.. table:: Approximately equivalent geometry structure terminology for
   Celeritas, Geant4, VecGeom, and SCALE.

   +------------------+-------------------------+----------------+--------------------+
   | Celeritas        | Geant4                  | VecGeom        | KENO/SCALE [#sc]_  |
   +==================+=========================+================+====================+
   | Object [#ob]_    | Solid                   | Unplaced       | Shape              |
   +------------------+-------------------------+----------------+--------------------+
   | Volume           | Logical volume  [#lv]_  | Logical volume | Unit/array/media   |
   +------------------+-------------------------+----------------+--------------------+
   | Volume instance  | Physical volume [#cn]_  | Placed volume  | Hole/array element |
   +------------------+-------------------------+----------------+--------------------+
   | Child            | Daughter                | Daughter       | Hole/placement     |
   +------------------+-------------------------+----------------+--------------------+
   | Parent           | Mother                  | Mother         | ---                |
   +------------------+-------------------------+----------------+--------------------+
   | Interface        | Border surface          | ---            | ---                |
   +------------------+-------------------------+----------------+--------------------+
   | Boundary         | Skin surface            | ---            | ---                |
   +------------------+-------------------------+----------------+--------------------+
   | Surface          | Logical surface         | ---            | ---                |
   +------------------+-------------------------+----------------+--------------------+

.. [#sc] The KENO geometry package in SCALE :cite:`scale-632` differs
   substantially from Geant4 geometry definitions. In KENO-VI :cite:`kenovi`
   geometry, parent units mask (rather than strictly contain) child units.

.. [#ob] :ref:`api_orange_objects` are used strictly for ORANGE construction in
   ORANGE and are not identifiable during runtime.

.. [#lv] The ``G4LogicalVolume`` class is equivalent to a volume except that
   points inside the children are not considered to be part of the volume. The
   same is true for ``VecGeom::LogicalVolume``.

.. [#cn] One ``G4PVReplica`` volume is expanded into *several* volume
   instances, one per "multiplicity".

Celeritas defines abstract geometry concepts, indexed as IDs, to support
multiple geometry applications [#ga]_ and to make the code backend-agnostic for
integrating with physics. These include "volumes" (known in some other
fields as "cells") and "surfaces" defined by the relationships between volumes.

.. [#ga] In the future the use of these abstract concepts will enable detector
   descriptions, and geometry models for other applications, that are *not*
   Geant4 hierarchies.



Volume
   A *volume* corresponds to a homogeneous physical object that can have multiple
   instances but is treated identically.
   It has a specific shape, material, metadata, and associated scoring/sensitive region.
   Each volume is simply a *node* in the detector geometry graph; it does not
   include its children.
   The Celeritas codebase sometimes refers to user-defined volumes as
   *canonical* to differentiate them from *implementation* volumes.

Volume instance
   An *instance* of a volume is defined in conjunction with a transform and an
   enclosing object (or, in the special case of the outermost or "world" volume
   instance, no enclosing object).
   In Geant4 this roughly corresponds to a "physical volume," except that
   Celeritas generates distinct volume instance IDs for every copy provided by
   a ``G4PVReplica``.
   VecGeom refers to volume instances as *placed volumes*.
   In a user-provided (or SCALE-generated) ORANGE geometry, a volume
   instance might correspond to a particular hole placement (i.e., a universe
   daughter), an array element, or a media entry (i.e., a cell).
   The volume instance is an *edge* in the graph of volumes.

Unique instance
   A *unique* instance of a volume refers to the logical definition of a
   specific region of global space in the geometry model. It is the full
   directed path :cite:`bender-listsdecisions-2010` from the root volume node
   (world volume) to a node (logical volume) somewhere in the graph, thereby
   describing all enclosing volumes and their locations. This path can be
   encoded uniquely as a single integer by pre-calculating the number of direct
   and indirect children for each node.  Celeritas always uses 64-bit integers
   to store the ``VolumeUniqueInstanceId``.

Surface
   A *surface* is defined as a contiguous area on the boundary of a volume,
   sometimes on only a single side of the volume. Note that this definition
   differs from the infinite surfaces of ORANGE and the surface frames of
   VecGeom. Surfaces currently are defined in two ways: *Interface* surfaces
   ("border" surfaces in Geant4) are one-directional surfaces defined as the
   interface from one volume instance to another. *Boundary* surfaces ("skin"
   surfaces in Geant4) surround an entire volume, and their properties apply
   symmetrically to tracks entering or exiting.

ImplVolume
   An *implementation volume* is a low-level detail used by each separate
   geometry representation/navigation implementation. They may correspond
   more or less to volumes, volume instances, unique volume instances, or
   anywhere in between. Use of these outside an individual geometry is
   discouraged; some parts of the code use these to map between Geant4
   pointers and the geometry state. Use the volume/instance/unique volume
   instead, relying on the cpp:class:`celeritas::VolumeParams` and global
   :cpp:class:`celeritas::GeantGeoParams` to convert between the local geometry
   state and the Geant4 navigation.


.. toctree::
   :maxdepth: 2

   geometry/interfaces.rst
   geometry/geant4.rst
   geometry/orange.rst
   geometry/vecgeom.rst
