.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_g4_interface:

Geant4 interface
================

The ``accel`` directory contains components exclusive to coupling Celeritas with
Geant4 for user-oriented integration. A simple interface for multithreaded or
serial applications is demonstrated in :ref:`example_geant`, and the more
advanced implementation can be inspected in the :ref:`celer-g4` app.

This integration interface is based on two main mechanisms:

* :cpp:class:`celeritas::UserActionIntegration`: Compatible with all Geant4
  versions supported by Celeritas.

* :cpp:class:`celeritas::TrackingManagerIntegration`: Compatible only with
  **Geant4 v11 and newer**, as it requires Geant4's
  :cpp:class:`G4VTrackingManager` interface for particles.

On either case, :cpp:class:`celeritas::SetupOptions` must be valid at the latest
during the ``G4UserRunAction::BeginOfRunAction()`` call in the **master**
thread, to initialize :cpp:class:`celeritas::SharedParams` and
:cpp:class:`celeritas::LocalTransporter`. Nevertheless, since the user might
need access to these options during other ``UserAction`` constructors (e.g.
initializing sensitive detector options in
:cpp:class:`G4VUserDetectorConstruction`), assigning them via
``celeritas::TrackingManagerIntegration::SetOptions`` before
``G4RunManager::SetUserInitialization`` calls is recommended.

.. toctree::
   :maxdepth: 2

   geant4-interface/high-level.rst
   geant4-interface/setup.rst
   geant4-interface/details.rst
   geant4-interface/low-level.rst
