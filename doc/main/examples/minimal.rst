.. Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _example_minimal:

Minimal Celeritas usage
=======================

This simple example shows how to incorporate an already-installed Celeritas
into a downstream project.

CMake infrastructure
--------------------

A three-line helper script ``FindCeleritas.cmake`` gives more verbose loading
messages when using Celeritas externally.

.. literalinclude:: ../../../example/FindCeleritas.cmake
   :language: cmake
   :start-at: find_package(

The CMake code itself is straightforward, though note the use of
``celeritas_target_link_libraries`` instead of ``target_link_libraries`` to
support CUDA RDC, which is required by VecGeom.

.. literalinclude:: ../../../example/minimal/CMakeLists.txt
   :language: cmake
   :start-at: project(

Main executable
---------------

.. literalinclude:: ../../../example/minimal/minimal.cc
   :start-at: #include

