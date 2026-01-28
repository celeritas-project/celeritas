.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

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
contract". They all throw :cpp:class:`celeritas::DebugError`.

.. doxygendefine:: CELER_EXPECT
.. doxygendefine:: CELER_ASSERT
.. doxygendefine:: CELER_ENSURE

The following two macros will throw debug assertions *or* cause undefined
behavior at runtime to allow compiler optimizations:

.. doxygendefine:: CELER_ASSERT_UNREACHABLE
.. doxygendefine:: CELER_ASSUME

Finally, a few runtime macros will always throw helpful
:cpp:class:`celeritas::RuntimeError` errors based on incorrect configuration or
input values.

.. doxygendefine:: CELER_VALIDATE
.. doxygendefine:: CELER_NOT_CONFIGURED
.. doxygendefine:: CELER_NOT_IMPLEMENTED

These macros all throw subclasses of standard exceptions. See the developer
documentation for more details.

.. doxygenclass:: celeritas::DebugError
.. doxygenclass:: celeritas::RuntimeError


Utility macros
^^^^^^^^^^^^^^

The :file:`corecel/Macros.hh` file defines language and compiler abstraction
macro definitions.

.. doxygendefine:: CELER_DEFAULT_COPY_MOVE
.. doxygendefine:: CELER_DELETE_COPY_MOVE
.. doxygendefine:: CELER_DEFAULT_MOVE_DELETE_COPY

.. doxygendefine:: CELER_DISCARD

.. _exception_handling:

Exception handling
^^^^^^^^^^^^^^^^^^

During multithreaded execution or when called by a Geant4 simulation, it is
important not to throw C++ exceptions lest they be uncaught and result in an
immediate program abort. Celeritas provides macros and classes for managing
exceptions and providing detailed diagnostics.

.. doxygendefine:: CELER_TRY_HANDLE
.. doxygendefine:: CELER_TRY_HANDLE_CONTEXT
.. doxygenclass:: celeritas::RichContextException
.. doxygenclass:: celeritas::MultiExceptionHandler
.. doxygenfunction:: celeritas::log_and_rethrow
