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

The track view provides access to the thread-local state.

.. doxygenclass:: celeritas::GeoTrackInterface
   :members:
