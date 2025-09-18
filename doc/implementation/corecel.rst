.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_corecel:

Core functionality
==================

The ``corecel`` directory contains functionality shared by Celeritas and ORANGE
primarily pertaining to GPU abstractions.

Configuration
-------------

The :file:`corecel/Config.hh` configure file contains all-caps definitions of the
CMake configuration options as 0/1 defines so they can be used with ``if
constexpr`` and other C++ expressions. In addition, it defines external C strings
with configuration options such as key dependent library versions.
Finally, :file:`corecel/Version.hh` defines version numbers as preprocessor
definition, a set of integers, and a descriptive string. The external API of
Celeritas should depend almost exclusively on the version, not the configured
options.

The files :file:`celeritas_{config,version,cmake_strings,sys_config}.h` and
:file:`corecel/device_runtime_api.h` are deprecated aliases for
backward-compatibility.
.. deprecated:: v0.5

   These will be removed in v0.6.

.. doxygendefine:: CELERITAS_VERSION


Fundamentals
------------

Several high-level types are defined as aliases since they may change based on
configuration: for example, ``size_type`` is a 32-bit integer when building
with device code enabled, but is a 64-bit integer on other 64-bit systems.

.. doxygentypedef:: celeritas::size_type
.. doxygentypedef:: celeritas::real_type

.. _debug_assertions:

Debug assertions
^^^^^^^^^^^^^^^^

Celeritas exception types and assertions are defined in
:file:`corecel/Assert`. Debug assertions (see :ref:`coding_testing`) are only
enabled when the
``CELERITAS_DEBUG`` (host code) and ``CELERITAS_DEVICE_DEBUG`` (device code)
CMake configuration options are set.

The assertion macros ``CELER_EXPECT``, ``CELER_ASSERT``, and ``CELER_ENSURE``
correspond to "precondition contract", "internal assertion", and "postcondition
contract".

.. doxygendefine:: CELER_EXPECT
.. doxygendefine:: CELER_ASSERT
.. doxygendefine:: CELER_ENSURE

The following two macros will throw debug assertions *or* cause undefined
behavior at runtime to allow compiler optimizations:

.. doxygendefine:: CELER_ASSERT_UNREACHABLE
.. doxygendefine:: CELER_ASSUME

Finally, a few runtime macros will always throw helpful errors based on
incorrect configuration or input values.

.. doxygendefine:: CELER_VALIDATE
.. doxygendefine:: CELER_NOT_CONFIGURED
.. doxygendefine:: CELER_NOT_IMPLEMENTED

Utility macros
^^^^^^^^^^^^^^

The :file:`corecel/Macros.hh` file defines language and compiler abstraction
macro definitions.

.. doxygendefine:: CELER_TRY_HANDLE
.. doxygendefine:: CELER_TRY_HANDLE_CONTEXT

.. doxygendefine:: CELER_DEFAULT_COPY_MOVE
.. doxygendefine:: CELER_DELETE_COPY_MOVE
.. doxygendefine:: CELER_DEFAULT_MOVE_DELETE_COPY

.. doxygendefine:: CELER_DISCARD

Platform portability macros
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :file:`Macros.hh` file also defines language and compiler abstraction macro
definitions.  It includes cross-platform (CUDA, C++, HIP) macros that expand to
attributes depending on the compiler and build configuration.

.. doxygendefine:: CELER_FUNCTION
.. doxygendefine:: CELER_CONSTEXPR_FUNCTION
.. doxygendefine:: CELER_DEVICE_COMPILE

The :file:`DeviceRuntimeApi` file, which must be included from all ``.cu``
files and ``.cc`` file which make CUDA/HIP API calls (see
:ref:`device_compilation`), provides cross-platform compatibility macros for
building against CUDA and HIP.

.. doxygendefine:: CELER_DEVICE_API_SYMBOL

An assertion macro in :file:`Assert.hh` checks the return result of CUDA/HIP API calls and throws a detailed exception if they fail:

.. doxygendefine:: CELER_DEVICE_API_CALL


.. _api_system:

System
------

The system subdirectory provides uniform interfaces to hardware and the
operating system.

GPU management
^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::Device
.. doxygenfunction:: celeritas::device
.. doxygenfunction:: celeritas::activate_device()

Environment variables
^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::Environment
.. doxygenfunction:: celeritas::environment
.. doxygenfunction:: celeritas::getenv
.. doxygenfunction:: celeritas::getenv_flag

MPI support
^^^^^^^^^^^

.. doxygenclass:: celeritas::ScopedMpiInit
.. doxygenclass:: celeritas::MpiCommunicator

Performance profiling
^^^^^^^^^^^^^^^^^^^^^

These classes generalize the different low-level profiling libraries, both
device and host, described in :ref:`profiling`.


.. doxygenfunction:: celeritas::use_profiling
.. doxygenclass:: celeritas::ScopedProfiling

Utility functions
-----------------

These functions replace or extend those in the C++ standard library
``<utility>`` header but work in GPU code without the
special ``--expt-relaxed-constexpr`` flag.

.. doxygenfunction:: celeritas::forward
.. doxygenfunction:: celeritas::move
.. doxygenfunction:: celeritas::trivial_swap
.. doxygenfunction:: celeritas::exchange

Algorithms
----------

These device-compatible functions replace or extend those in the C++ standard
library ``<algorithm>`` header. The implementations of ``sort`` and other
partitioning elements are derived from LLVM's ``libc++``.

.. doxygenfunction:: celeritas::all_of
.. doxygenfunction:: celeritas::any_of
.. doxygenfunction:: celeritas::all_adjacent
.. doxygenfunction:: celeritas::lower_bound
.. doxygenfunction:: celeritas::lower_bound_linear
.. doxygenfunction:: celeritas::upper_bound
.. doxygenfunction:: celeritas::find_sorted
.. doxygenfunction:: celeritas::partition
.. doxygenfunction:: celeritas::sort
.. doxygenfunction:: celeritas::max
.. doxygenfunction:: celeritas::min
.. doxygenfunction:: celeritas::min_element

Numerics
--------

These functions replace or extend those in the C++ standard library
``<cmath>`` and ``<numeric>`` headers.

.. doxygenfunction:: celeritas::clamp
.. doxygenfunction:: celeritas::clamp_to_nonneg
.. doxygenfunction:: celeritas::ipow
.. doxygenfunction:: celeritas::fastpow
.. doxygenfunction:: celeritas::rsqrt(double)
.. doxygenfunction:: celeritas::fma
.. doxygenfunction:: celeritas::ceil_div
.. doxygenfunction:: celeritas::negate
.. doxygenfunction:: celeritas::eumod
.. doxygenfunction:: celeritas::signum
.. doxygenfunction:: celeritas::sincos(double a, double* s, double* c)
.. doxygenfunction:: celeritas::sincospi(double a, double* s, double* c)
.. doxygenfunction:: celeritas::popcount

.. doxygenstruct:: celeritas::numeric_limits
   :members:

Atomics
--------

These atomic functions are for use in kernel code (CUDA/HIP/OpenMP) that use
track-level parallelism.

.. doxygenfunction:: celeritas::atomic_add
.. doxygenfunction:: celeritas::atomic_min
.. doxygenfunction:: celeritas::atomic_max

Array utilities
---------------

These operate on fixed-size arrays of data (see :ref:`api_containers`), usually ``Real3`` as a
Cartesian spatial coordinate.

.. doxygentypedef:: celeritas::Real3

.. doxygenfunction:: celeritas::axpy
.. doxygenfunction:: celeritas::dot_product
.. doxygenfunction:: celeritas::cross_product
.. doxygenfunction:: celeritas::norm(Array<T, N> const &v)
.. doxygenfunction:: celeritas::make_unit_vector
.. doxygenfunction:: celeritas::distance
.. doxygenfunction:: celeritas::from_spherical
.. doxygenfunction:: celeritas::rotate


Soft equivalence
----------------

These utilities are used for comparing real-valued numbers to a given
tolerance.

.. doxygenclass:: celeritas::SoftEqual
.. doxygenclass:: celeritas::SoftZero
.. doxygenclass:: celeritas::EqualOr
.. doxygenclass:: celeritas::ArraySoftUnit


.. _api_io:

I/O
---

These functions and classes are for communicating helpfully with the user.

.. doxygendefine:: CELER_LOG
.. doxygendefine:: CELER_LOG_LOCAL
.. doxygenenum:: celeritas::LogLevel
   :no-link:
