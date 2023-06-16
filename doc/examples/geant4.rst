.. Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _example_geant:

Minimal Geant4 integration
==========================

This small example demonstrates how to offload tracks to Celeritas in a serial
or multithreaded environment. The :ref:`accel` library is the only part of
Celeritas that needs to be understood for it to work. The key components are
global SetupOptions and SharedParams, coupled to thread-local SimpleOffload and
LocalTransporter. The SimpleOffload provides all the methods needed to
integrate into Geant4 application's UserActions.

.. _example_cmake:

CMake infrastructure
--------------------

.. literalinclude:: ../../example/accel/CMakeLists.txt
   :language: cmake
   :start-at: project(

Main executable
---------------

This single executable is a less robust (and minimally documented) version of
the :ref:`celer-g4` app. Its use of global variables rather than shared
pointers is easier to implement but may be more problematic with experiment
frameworks or other apps that use a task-based runner.

.. literalinclude:: ../../example/accel/accel.cc
   :start-at: #include

