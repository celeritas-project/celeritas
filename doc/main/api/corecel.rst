.. Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_corecel:

Core package
============

The ``corecel`` directory contains functionality shared by Celeritas and ORANGE
primarily pertaining to GPU abstractions.

Configuration
-------------

The ``celeritas_config.h`` configure file contains all-caps definitions of the
CMake configuration options as 0/1 defines so they can be used with ``if
constexpr`` and other C++ expressions. The ``celeritas_cmake_strings.h``
defines static C strings with configuration options such as key dependent
library versions. Finally, ``celeritas_version.h`` defines version numbers as
a preprocessor definition, a set of integers, and a descriptive string.

.. doxygendefine:: CELERITAS_VERSION
.. doxygenvariable:: celeritas_version

Fundamentals
------------

.. doxygenfile:: corecel/Macros.hh

Celeritas assertions are only enabled when the ``CELERITAS_DEBUG``
configuration option is set. The macros ``CELER_EXPECT``, ``CELER_ASSERT``, and
``CELER_ENSURE`` correspond to "precondition contract", "internal assertion",
and "postcondition contract".

.. doxygenfile:: corecel/Assert.hh

.. doxygenfile:: corecel/Types.hh

.. doxygenclass:: celeritas::OpaqueId

System
------

.. doxygenclass:: celeritas::Device
.. doxygenfunction:: celeritas::device
.. doxygenfunction:: celeritas::activate_device()

Containers
----------

.. doxygenstruct:: celeritas::Array
   :undoc-members:

.. doxygenclass:: celeritas::Span


Math, numerics, and algorithms
------------------------------

.. doxygenfile:: corecel/math/Algorithms.hh

.. doxygenfile:: corecel/math/ArrayUtils.hh

.. doxygenfile:: corecel/math/Atomics.hh

.. doxygenstruct:: celeritas::numeric_limits

.. _api_quantity:

.. doxygenfile:: corecel/math/Quantity.hh

.. doxygenclass:: celeritas::SoftEqual


.. _api_io:

I/O
---

.. doxygenfile:: corecel/io/Logger.hh

.. doxygenclass:: celeritas::OutputInterface

Data
----

Data *storage* must be isolated from data *use* for any code that is to run on
the device. This
allows low-level physics classes to operate on references to data using the
exact same device/host code. Furthermore, state data (one per track) and
shared data (definitions, persistent data, model data) should be separately
allocated and managed.

Params
  Provide a CPU-based interface to manage and provide access to constant shared
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
   reside in ``ParticleParams``, and the particle track properties are managed
   by a ``ParticleStateStore`` class.

   A separate class, the ``ParticleTrackView``, is instantiated with a
   specific thread ID so that it acts as an accessor to the
   stored data for a particular track. It can calculate properties that depend
   on both the state and parameters. For example, momentum depends on both the
   mass of a particle (constant, set by the model) and the speed (variable,
   depends on particle track state).

Users and other parts of the code can add their own shared and stream-local
(i.e., thread-local) data to Celeritas using the ``AuxParamsInterface`` and ``AuxStateInterface`` classes, accessed through the  ``AuxParamsRegistry`` and ``AuxStateVec`` classes, respectively.

.. doxygenclass:: celeritas::AuxParamsInterface

.. doxygenclass:: celeritas::AuxParamsRegistry

.. doxygenclass:: celeritas::AuxStateVec


