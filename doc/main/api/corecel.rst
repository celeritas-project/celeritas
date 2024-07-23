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

Several high-level types are defined as aliases since they may change based on
configuration: for example, ``size_type`` is a 32-bit integer when building
with device code enabled, but is a 64-bit integer on other 64-bit systems.

.. doxygentypedef:: celeritas::size_type
.. doxygentypedef:: celeritas::real_type

Macros
~~~~~~

The :file:`Macros.hh` file defines language and compiler abstraction macro
definitions.  It includes cross-platform (CUDA, C++, HIP) macros that expand to
attributes depending on the compiler and build configuration.

.. doxygendefine:: CELER_FUNCTION
.. doxygendefine:: CELER_CONSTEXPR_FUNCTION
.. doxygendefine:: CELER_DEVICE_COMPILE
.. doxygendefine:: CELER_TRY_HANDLE
.. doxygendefine:: CELER_TRY_HANDLE_CONTEXT
.. doxygendefine:: CELER_DEFAULT_COPY_MOVE
.. doxygendefine:: CELER_DELETE_COPY_MOVE
.. doxygendefine:: CELER_DEFAULT_MOVE_DELETE_COPY
.. doxygendefine:: CELER_DISCARD

Debug assertions
~~~~~~~~~~~~~~~~

Celeritas debug assertions are only enabled when the ``CELERITAS_DEBUG``
configuration option is set. The macros ``CELER_EXPECT``, ``CELER_ASSERT``, and
``CELER_ENSURE`` correspond to "precondition contract", "internal assertion",
and "postcondition contract".

.. doxygendefine:: CELER_EXPECT
.. doxygendefine:: CELER_ASSERT
.. doxygendefine:: CELER_ENSURE

The following two macros will throw debug assertions *or* cause undefined
behavior at runtime:

.. doxygendefine:: CELER_ASSERT_UNREACHABLE
.. doxygendefine:: CELER_ASSUME

Finally, a few runtime macros will always throw helpful errors based on
incorrect configuration or input values.

.. doxygendefine:: CELER_VALIDATE
.. doxygendefine:: CELER_NOT_CONFIGURED
.. doxygendefine:: CELER_NOT_IMPLEMENTED

.. _api_system:

System
------

.. doxygenclass:: celeritas::Device
.. doxygenfunction:: celeritas::device
.. doxygenfunction:: celeritas::activate_device()

.. doxygenclass:: celeritas::Environment
.. doxygenfunction:: celeritas::environment
.. doxygenfunction:: celeritas::getenv
.. doxygenfunction:: celeritas::getenv_flag

Containers
----------

.. doxygenstruct:: celeritas::Array

.. doxygenclass:: celeritas::Span


Math, numerics, and algorithms
------------------------------

.. doxygenfile:: corecel/math/Algorithms.hh

.. doxygenfile:: corecel/math/ArrayUtils.hh

.. doxygenfile:: corecel/math/Atomics.hh

.. doxygenstruct:: celeritas::numeric_limits
   :members:

.. doxygenclass:: celeritas::SoftEqual

.. _api_quantity:

.. doxygenclass:: celeritas::Quantity
.. doxygenfunction:: celeritas::native_value_to
.. doxygenfunction:: celeritas::native_value_from(Quantity<UnitT, ValueT> quant)
.. doxygenfunction:: celeritas::value_as
.. doxygenfunction:: celeritas::zero_quantity
.. doxygenfunction:: celeritas::max_quantity
.. doxygenfunction:: celeritas::neg_max_quantity


.. _api_io:

I/O
---

These functions and classes are for communicating helpfully with the user.

.. doxygendefine:: CELER_LOG
.. doxygendefine:: CELER_LOG_LOCAL
.. doxygenenum:: celeritas::LogLevel

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

Storage
~~~~~~~

.. doxygenpage:: collections

.. doxygenenum:: celeritas::MemSpace
.. doxygenenum:: celeritas::Ownership

.. doxygenclass:: celeritas::OpaqueId

.. doxygentypedef:: celeritas::ItemId
.. doxygentypedef:: celeritas::ItemRange
.. doxygenclass:: celeritas::ItemMap

.. doxygenclass:: celeritas::Collection
.. doxygenclass:: celeritas::CollectionMirror

Auxiliary user data
~~~~~~~~~~~~~~~~~~~

Users and other parts of the code can add their own shared and stream-local
(i.e., thread-local) data to Celeritas using the ``AuxParamsInterface`` and ``AuxStateInterface`` classes, accessed through the  ``AuxParamsRegistry`` and ``AuxStateVec`` classes, respectively.

.. doxygenclass:: celeritas::AuxParamsInterface

.. doxygenclass:: celeritas::AuxParamsRegistry

.. doxygenclass:: celeritas::AuxStateVec
