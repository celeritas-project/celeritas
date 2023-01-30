.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_corecel:

Core package
============

The ``corecel`` directory contains functionality shared by Celeritas and ORANGE
primarily pertaining to GPU abstractions.

Configuration
-------------

The configure file contains all-caps definitions of the CMake configuration
options as 0/1 defines.

.. doxygenfile:: celeritas_config.h

.. doxygenfile:: celeritas_cmake_strings.h

.. doxygenfile:: celeritas_version.h

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

.. doxygenfile:: corecel/sys/Device.hh

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

.. doxygenfile:: corecel/math/SoftEqual.hh

I/O
---

.. doxygenfile:: corecel/io/Logger.hh

