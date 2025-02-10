.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _example_template:

.. include:: ../../example/offload-template/README.rst

.. _example_geant:

Geant4 integration examples
===========================

These small examples demonstrate how to offload tracks to Celeritas in a serial
or multithreaded environment using:

#. A concrete G4VTrackingManager
#. A concrete G4UserTrackingAction user action class
#. A concrete G4VFastSimulationModel

The :ref:`api_g4_interface` is the only relevant part of Celeritas here.
The key components are global :cpp:struct:`celeritas::SetupOptions` and
:cpp:class:`celeritas::SharedParams`, integrated with
:cpp:class:`celeritas::TrackingManagerIntegration`.

.. todo::
   The deprecated SimpleOffload class is used in two of these examples. Update
   them using the newer integration and add description for user action
   integration when implemented.

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

Offload using a concrete G4VTrackingManager
-------------------------------------------

The tracking manager is the preferred way of offloading tracks to Celeritas. It
requires Geant4 11.0 or higher.

.. literalinclude:: ../../example/accel/trackingmanager-offload.cc
   :start-at: #include

Offload using a concrete G4UserTrackingAction
---------------------------------------------

.. literalinclude:: ../../example/accel/simple-offload.cc
   :start-at: #include

Offload using a concrete G4VFastSimulationModel
-----------------------------------------------

.. literalinclude:: ../../example/accel/fastsim-offload.cc
   :start-at: #include
