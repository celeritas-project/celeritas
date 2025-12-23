.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _system:

.. _api_system:

System
------

The system subdirectory provides uniform interfaces to hardware and the
operating system.

Configuration
^^^^^^^^^^^^^

The :file:`corecel/Config.hh` configure file contains all-caps definitions of the
CMake configuration options as 0/1 defines so they can be used with ``if
constexpr`` and other C++ expressions. In addition, it defines external C strings
with configuration options such as key dependent library versions.

Additionally, :file:`corecel/Version.hh` defines version numbers as preprocessor
definition, a set of integers, and a descriptive string. The external API of
Celeritas should depend almost exclusively on the version, not the configured
options.

.. doxygendefine:: CELERITAS_VERSION

GPU management
^^^^^^^^^^^^^^

.. doxygenclass:: celeritas::Device
.. doxygenfunction:: celeritas::device
.. doxygenfunction:: celeritas::activate_device()


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


.. doxygenclass:: celeritas::ScopedProfiling
.. doxygenclass:: celeritas::TracingSession


.. _api_io:

I/O
^^^

These functions and classes are for communicating helpfully with the user.

.. doxygendefine:: CELER_LOG
.. doxygendefine:: CELER_LOG_LOCAL
.. doxygenenum:: celeritas::LogLevel
   :no-link:

.. doxygenclass:: celeritas::Logger
.. doxygenclass:: celeritas::ScopedSignalHandler
