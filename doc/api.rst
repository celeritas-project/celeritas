.. Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _api:

*****************
API documentation
*****************

.. only:: nobreathe

   .. note:: The breathe_ extension was not used when building this version of
      the documentation. The API documentation will not be rendered below.

   .. _breathe: https://github.com/michaeljones/breathe#readme

The Celeritas codebase lives under the ``src/`` directory and is divided into
three packages. Additional top-level files provide access to version and
configuration attributes.

.. doxygenfile:: celeritas_version.h

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

.. doxygenclass:: celeritas::Array
   :undoc-members:

.. doxygenfile:: corecel/cont/Range.hh

.. doxygenfile:: corecel/cont/Span.hh


Math, numerics, and algorithms
------------------------------

.. doxygenfile:: corecel/math/Algorithms.hh

.. doxygenfile:: corecel/math/ArrayUtils.hh

.. doxygenfile:: corecel/math/Atomics.hh

.. doxygenfile:: corecel/math/NumericLimits.hh

.. doxygenfile:: corecel/math/Quantity.hh

.. doxygenfile:: corecel/math/SoftEqual.hh

I/O
---

.. doxygenfile:: corecel/io/Logger.hh

ORANGE
======

The ORANGE (Oak Ridge Advanced Nested Geometry Engine) package is currently
under development as the version in SCALE is ported to GPU.

.. doxygenclass:: celeritas::OrangeParams

.. doxygenclass:: celeritas::OrangeTrackView

Celeritas
=========

Problem definition
------------------

.. doxygenclass:: celeritas::MaterialParams

.. doxygenclass:: celeritas::ParticleParams

.. doxygenclass:: celeritas::PhysicsParams

Transport interface
-------------------

.. doxygenclass:: celeritas::Stepper

On-device access
----------------

.. doxygenclass:: celeritas::MaterialTrackView

.. doxygenclass:: celeritas::ParticleTrackView

.. doxygenclass:: celeritas::PhysicsTrackView

