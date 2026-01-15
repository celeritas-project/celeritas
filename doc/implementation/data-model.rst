.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_data_model:

Data model
==========

Data *storage* must be isolated from data *use* for any code that is to run on
the device. This
allows low-level physics classes to operate on references to data using the
exact same device/host code. Furthermore, state data (one per track) and
shared data (definitions, persistent data, model data) should be separately
allocated and managed.

Params
  Provide a host-side interface to manage and provide access to constant shared
  GPU data, usually model parameters or the like. The Params class itself can
  only be accessed via host code. A params class can contain metadata (string
  names, etc.) suitable for host-side debug output and for helping related
  classes convert from user-friendly input (e.g. particle name) to
  device-friendly IDs (e.g., particle ID). These classes should inherit from
  the ``ParamsDataInterface`` class to define uniform helper methods and types
  and will often implement the data storage by using ``CollectionMirror``.

State
  Thread-local data specifying the state of a single particle track with
  respect to a corresponding params class (``FooParams``). In the main
  Celeritas stepping loop, all state data is managed via the ``CoreState``
  class.

View
  Device-friendly class that provides read and/or write access to shared and
  local state data. The name is in the spirit of
  ``std::string_view``, which adds functionality to non-owned data.
  It combines the state variables and model
  parameters into a single class. The constructor always takes const references
  to ParamsData and StateData as well as the track slot ID. It encapsulates
  the storage/layout of the state and parameters, as well as what (if any) data
  is cached in the state.

.. hint::

   Consider the following example.

   All SM physics particles share a common set of properties such as mass and
   charge, and each instance of particle has a particular set of
   associated variables such as kinetic energy. The shared data (SM parameters)
   reside in ``ParticleParams``, and the particle track properties are stored
   as part of the core state.

   A separate class, the ``ParticleTrackView``, is instantiated with a
   specific thread ID so that it acts as an accessor to the
   stored data for a particular track. It can calculate properties that depend
   on both the state and parameters. For example, momentum depends on both the
   mass of a particle (constant, set by the model) and the speed (variable,
   depends on particle track state).

Storage
-------

.. doxygenpage:: collections

.. doxygenenum:: celeritas::MemSpace
   :no-link:
.. doxygenenum:: celeritas::Ownership
   :no-link:

.. doxygenclass:: celeritas::OpaqueId
.. doxygenfunction:: celeritas::id_cast

.. doxygentypedef:: celeritas::ItemId
.. doxygentypedef:: celeritas::ItemRange
.. doxygenclass:: celeritas::ItemMap

.. doxygenclass:: celeritas::Collection
.. doxygenclass:: celeritas::CollectionMirror
.. doxygenclass:: celeritas::CollectionStateStore

.. doxygenfunction:: celeritas::ldg

.. _api_containers:

Containers
----------

These are containers and container-like objects used throughout Celeritas.

.. doxygenclass:: celeritas::Array
.. doxygenclass:: celeritas::EnumArray
.. doxygenclass:: celeritas::Range
.. doxygenclass:: celeritas::Span


.. _api_auxiliary_data:

Auxiliary storage
-----------------

Users and other parts of the code can add their own shared and stream-local
(i.e., thread-local) data to Celeritas using the
:cpp:class:`celeritas::AuxParamsInterface` and
:cpp:class:`celeritas::AuxStateInterface` classes, accessed through the
:cpp:class:`celeritas::AuxParamsRegistry` and
:cpp:class:`celeritas::AuxStateVec` classes, respectively.

.. doxygenclass:: celeritas::AuxParamsInterface
.. doxygenclass:: celeritas::AuxStateInterface
.. doxygenclass:: celeritas::AuxParamsRegistry
.. doxygenclass:: celeritas::AuxStateVec

Auxiliary collection groups
---------------------------

.. doxygenclass:: celeritas::AuxParams
