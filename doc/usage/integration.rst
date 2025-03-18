.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. highlight:: cmake

.. _library:

Software library
================

The most stable part of Celeritas is, at the present time, the high-level
:ref:`api_g4_interface`. However, many other
components of the API are stable and documented in the :code:`api` section.

CMake integration
-----------------

The Celeritas library is most easily used when your downstream app is built with
CMake, demonstrated by the :ref:`example_minimal` and :ref:`example_cmake` examples. It should require a single line to initialize::

   find_package(Celeritas REQUIRED)

and a single line to link::

   cuda_rdc_target_link_libraries(mycode PUBLIC Celeritas::celeritas)

CudaRdcUtils
^^^^^^^^^^^^

The special ``cuda_rdc_`` command in the previous statement,
rather than the simpler native CMake command
``target_link_libraries``, is needed in case Celeritas is built with both
CUDA and the current version of VecGeom, which uses a special but messy feature
called CUDA Relocatable Device Code (RDC).
As the ``cuda_rdc_...`` functions decay to the wrapped CMake commands if CUDA
and VecGeom are disabled, you can use them to safely build and link nearly all targets
consuming Celeritas in your project. This provides tracking of the appropriate
sequence of linking for the final application whether it uses CUDA code or not,
and whether Celeritas is CPU-only or CUDA enabled::

  cuda_rdc_add_library(myconsumer SHARED ...)
  cuda_rdc_target_link_libraries(myconsumer PUBLIC Celeritas::celeritas)

  cuda_rdc_add_executable(myapplication ...)
  cuda_rdc_target_link_libraries(myapplication PRIVATE myconsumer)

.. note:: Prior to Celeritas v0.6, obtaining the above commands required the
   user to include :file:`CudaRdcUtils.cmake` manually. Now, it is included
   automatically.

If your project builds shared libraries that are intended to be loaded at
application runtime (e.g. via ``dlopen``), you should prefer use the CMake
``MODULE`` target type::

  cuda_rdc_add_library(myplugin MODULE ...)
  cuda_rdc_target_link_libraries(myplugin PRIVATE Celeritas::celeritas)

This is recommended as ``cuda_rdc_target_link_libraries`` understands these as
a final target for which all device symbols require resolving. If you are
forced to use the ``SHARED`` target type for plugin libraries (e.g. via your
project's own wrapper functions), then these should be declared with the CMake
or project-specific commands with linking to both the primary Celeritas target
and its device code counterpart::

  add_library(mybadplugin SHARED ...)
  # ... or myproject_add_library(mybadplugin ...)
  target_link_libraries(mybadplugin
    PRIVATE Celeritas::celeritas $<TARGET_NAME_IF_EXISTS:Celeritas::celeritas_final>
  )
  # ... or otherwise declare the plugin as requiring linking to the two targets

Celeritas device code counterpart target names are always the name of the
primary target appended with ``_final``. They are only present if Celeritas was
built with CUDA support so it is recommended to use the CMake generator
expression above to support CUDA or CPU-only builds transparently.

Targets
^^^^^^^

CMake targets exported by Celeritas live in the ``Celeritas::`` namespace.
These targets are:

- An interface lib ``BuildFlags`` that provides the include path to Celeritas,
  language requirements, and warning overrides;
- Wrapper interface libraries that provide links to optional dependencies if
  enabled: ``ExtMPI``, ``ExtOpenMP``, ``ExtPerfetto``, ``ExtGeant4Geo``
  (G4Geometry if available), and ``ExtDeviceApi`` (CUDA::toolkit if available,
  ROCM/HIP libraries and includes, and roctx64 if available);
- Code libraries described in :ref:`api`: corecel, geocel, orange,
  celeritas, and accel;
- Executables such as ``celer-sim`` and ``celer-geo``.


App integration
---------------

Integrating with Geant4 user applications and experiment frameworks requires
setting up:

- Initialization options that depend on the problem being run and the system
  architecture (see :cpp:class:`celeritas::SetupOptions` until version 1.0, or
  :ref:`input` after)
- "Global" data structures for shared problem data such as geometry (see
  :cpp:class:`celeritas::SharedParams`)
- Per-worker (i.e., ``G4ThreadLocal`` using Geant4 manager/worker semantics)
  data structures that hold the track states (see :cpp:class:`celeritas::LocalTransporter`).

See :ref:`example_geant` for examples of integrating Geant4, and :ref:`api_g4_interface` for detailed documentation of the Geant4 integration interface.
