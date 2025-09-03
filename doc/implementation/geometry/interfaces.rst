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
