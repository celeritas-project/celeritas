.. Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _example_geant:

Minimal Geant4 integration
==========================

These small examples demonstrate how to offload tracks to Celeritas in a serial
or multithreaded environment using:

#. A concrete G4UserTrackingAction user action class
#. A concrete G4VFastSimulationModel
#. A concrete G4VTrackingManager

The :ref:`api_g4_interface` is the only relevant part of Celeritas here.
The key components are global SetupOptions and SharedParams, coupled to thread-local
SimpleOffload and LocalTransporter. The SimpleOffload provides all of the core
methods needed to integrate into a Geant4 application's UserActions or other user classes.

.. _example_cmake:

CMake infrastructure
--------------------

.. literalinclude:: ../../example/accel/CMakeLists.txt
   :language: cmake
   :start-at: project(
   :end-before: END EXAMPLE CODE

Main Executables
----------------
The executables are less robust (and minimally documented) versions of
the :ref:`celer-g4` app. The use of global variables rather than shared
pointers is easier to implement but may be more problematic with experiment
frameworks or other apps that use a task-based runner.

Offload using a concrete G4UserTrackingAction
---------------------------------------------

.. literalinclude:: ../../example/accel/simple-offload.cc
   :start-at: #include

Offload using a concrete G4VFastSimulationModel
-----------------------------------------------

.. literalinclude:: ../../example/accel/fastsim-offload.cc
   :start-at: #include

Offload using a concrete G4VTrackingManager
--------------------------------------------

.. literalinclude:: ../../example/accel/trackingmanager-offload.cc
   :start-at: #include
