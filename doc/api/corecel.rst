.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. doxygenfile:: celeritas_version.h

.. _corecel:

Core package
============

The ``corecel`` directory contains functionality shared by Celeritas and ORANGE
primarily pertaining to GPU abstractions.

Fundamentals
------------

.. doxygenfile:: corecel/Macros.hh

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

.. doxygenfile:: corecel/math/Quantity.hh

.. doxygenfile:: corecel/math/SoftEqual.hh

I/O
---

.. doxygenfile:: corecel/io/Logger.hh

