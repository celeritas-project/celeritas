.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Interfaces
==========

These classes manage access to geometric information throughout the codebase.
They work in tandem with :cpp:class:`celeritas::GeantGeoParams` to associate
volume identifiers with Geant4 runtime data structures.

.. doxygenclass:: celeritas::VolumeParams

.. doxygenclass:: celeritas::SurfaceParams

.. doxygenclass:: celeritas::GeoParamsInterface

A few helper functions can be used to build collections (see
:ref:`api_data_model`) of ``ImplVolumeId`` for runtime tracking (used
internally by fields, physics, etc.).

.. doxygenfunction:: celeritas::build_volume_collection

.. doxygenclass:: celeritas::VolumeMapFiller


Runtime usage
-------------

A geometry "track view" is the key class used to access and/or modify a
geometry state. (See :ref:`api_data_model` for more discussion of views and
states.) Most states are a combination of physical properties (position,
direction) and logical properties (volume path, surface, surface side).

The track view provides mutators for changing the state:

- The assignment operator takes a :cpp:class:`celeritas::GeoTrackInitializer`
  object or a "detailed" (other state plus new direction) initializer to locate
  the point in the geometry hierarchy.
- ``move_to_boundary`` moves the track to the boundary of the current volume along
  the current direction, updating its logical state to indicate that it is on
  the boundary of the current volume
- ``cross_boundary`` changes the logical state when on the boundary, updating
  to the next volume
- ``move_internal`` changes the physical position of the geometry state without
  altering the logical state (i.e., it must remain within the current volume)
- ``set_dir`` changes the direction of the track (in global coordinate system)

Query operations return properties of the current state:

- ``pos`` and ``dir`` return the physical position and direction in the global
  coordinate system
- ``volume_id`` and ``volume_instance_id`` obtain canonical geometry
  information about the state
- ``is_outside`` returns whether the track has left the world (or started
  outside the outermost known volume)
- ``is_on_boundary`` is true if a track is exactly on the boundary of a volume,
  capable of changing to another volume without altering the physical position
- ``normal`` returns a global-coordinate direction perpendicular to the volume
  at the local point when on a boundary. The sign of the surface normal is
  implementation-dependent; it may change based on the track state (previous
  volume, direction, surface sign) or geometry construction.

Some operations return properties but update the internal state (e.g., setting
a "next volume" attribute).

- ``find_next_step`` determines the distance to the next boundary (i.e., a
  different implementation volume) along the track's current direction,
  optionally up to a given distance. Queries may be more efficient for small
  distances.
- ``find_safety`` determines the distance to the nearest boundary in any
  direction (i.e., the radius of the maximally inscribed sphere), up to an
  optional maximum distance.
