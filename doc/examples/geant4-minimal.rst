.. Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0


Minimal Geant4 integration
==========================

This small example demonstrates how to offload tracks to Celeritas.

.. _example_cmake:

CMake infrastructure
--------------------

.. literalinclude:: ../../example/accel/CMakeLists.txt
   :language: cmake
   :start-at: project(

Main executable
---------------

This single executable is a less robust (and minimally documented) version of
the larger :ref:`example_geant_full` example. Its use of global variables
rather than shared pointers is easier to implement but may be more problematic
with experiment frameworks or other apps that use a task-based runner.

.. literalinclude:: ../../example/accel/accel.cc
   :start-at: #include

