.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

Fundamentals
------------

Several high-level types are defined as aliases since they may change based on
configuration: for example, ``size_type`` is a 32-bit integer when building
with device code enabled, but is a 64-bit integer on other 64-bit systems.

.. doxygentypedef:: celeritas::size_type
.. doxygentypedef:: celeritas::real_type

.. doxygennamespace:: celeritas::literals

.. _debug_assertions:

Assertions and error checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Celeritas exception types and assertions are defined in
:file:`corecel/Assert.hh`.
Debug assertions (see :ref:`coding_testing`) are enabled through macros in :file:`corecel/Config.hh` defined through CMake:

.. doxygendefine:: CELERITAS_DEBUG
.. doxygendefine:: CELERITAS_DEVICE_DEBUG

All of the following macros are *statements* and should be followed by
a semicolon when used in code.

Errors relating to the system, configuration, or user input will throw
:cpp:class:`celeritas::RuntimeError` and are always enabled.
These cannot be used in device code: they are intended to be
called during setup time in host code.

.. doxygendefine:: CELER_VALIDATE
.. doxygendefine:: CELER_NOT_CONFIGURED
.. doxygendefine:: CELER_NOT_IMPLEMENTED

The debug assertion macros ``CELER_EXPECT``, ``CELER_ASSERT``, and
``CELER_ENSURE`` correspond to "precondition contract", "internal assertion",
and "postcondition contract".
They are checks for internal programming errors and (on host code) will throw
:cpp:class:`celeritas::DebugError` upon failing.

.. doxygendefine:: CELER_EXPECT
.. doxygendefine:: CELER_ASSERT
.. doxygendefine:: CELER_ENSURE

The following two macros will throw debug assertions *or* cause undefined
behavior at runtime to allow compiler optimizations:

.. doxygendefine:: CELER_ASSERT_UNREACHABLE
.. doxygendefine:: CELER_ASSUME

The debug (assertion failure) and runtime (validation/configuration failure)
errors are subclasses of standard library classes.

.. doxygenclass:: celeritas::DebugError
.. doxygenclass:: celeritas::RuntimeError

Some fairly complicated logic is in place to reuse code and ensure device
compatibility for all these macros (see :numref:`fig-assert`).

.. _fig-assert:

.. figure:: /_static/dot/assert.*
   :align: center
   :width: 90%

   Flow chart describing dispatches for the public API macros (blue) and
   ``[[noreturn]]`` functions (red), with implementation macros and functions
   (gray) in between.
   The "debug enabled" nodes use :c:macro:`CELERITAS_DEBUG` during host
   compilation or :c:macro:`CELERITAS_DEVICE_DEBUG` during device
   compilation.


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
immediate program abort
Celeritas provides macros and classes for managing exceptions and providing
detailed diagnostics.

.. doxygendefine:: CELER_TRY_HANDLE
.. doxygendefine:: CELER_TRY_HANDLE_CONTEXT
.. doxygenclass:: celeritas::RichContextException
.. doxygenclass:: celeritas::MultiExceptionHandler
.. doxygenfunction:: celeritas::log_and_rethrow
