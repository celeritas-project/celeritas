.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _api_g4_interface:

Geant4 interface
================

The ``accel`` directory contains components exclusive to coupling Celeritas with
Geant4 for user-oriented integration. A simple interface for multithreaded (MT)
or serial applications is demonstrated in :ref:`example_geant`, and the more
advanced implementation can be inspected in the :ref:`celer-g4` app.

Offloading through Celeritas is performed through two main classes,
:cpp:class:`celeritas::SharedParams`, which contains problem information shared
among all CPU and GPU threads, and :cpp:class:`celeritas::LocalTransporter`
which is associated to a single Geant4 event/task/thread. For most use cases,
a simpler integration interface should be used:

* :cpp:class:`celeritas::UserActionIntegration`: Compatible with all Geant4
  versions supported by Celeritas.

* :cpp:class:`celeritas::TrackingManagerIntegration`: Compatible only with
  **Geant4 version 11 and newer**, as it requires Geant4's
  :cpp:class:`G4VTrackingManager` interface for particles.

Running Geant4
--------------

Geant4's `run manager`_ imposes a particular ordering of execution:

1. The Geant4 run manager is created. Based on code input and/or the
   ``G4FORCE_RUN_MANAGER_TYPE`` environment variable, a
   serial/multithread/task-based run manager is built. Some experiment
   frameworks such as CMSSW have custom run managers that supplant Geant4's.
2. Initialization classes are provided to the run manager: these specify how
   the detector geometry and particles will be built on the main thread and how
   sensitive detectors, fields, and physics processes will be built on the worker
   threads. (In an MT run, ``UserActionInitialization::BuildForMaster``
   is called here. Particles are constructed immediately when the physics is
   provided to the run manager.)
3. The Geant4 "UI" (macro file) can be used to modify run parameters such as
   the number of threads, physics parameters, etc. The UI "messengers", created
   at various times before this, store pointers to long-lived data used during
   initialization, and macro commands directly update those values.
4. The run manager's ``Initialize`` method (which can also be called by the
   macro file) loads the detector geometry and sets up physics.
5. The run manager's ``BeamOn`` method (also callable by macro) starts the run.
6. Additional runs can also be performed afterward, sometimes with changes to
   physics and other properties in between.

.. _run manager: https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/Fundamentals/run.html#manage-the-run-procedures

During initialization:

1. The detector geometry is loaded on the main thread.
2. In an serial run, ``UserActionInitialization::Build`` is called on the
   thread.
3. Thread-local (on both main and worker) Geant4 components are constructed:
   sensitive detectors, fields, and physics processes in that order. At the end
   of physics initialization, tracking managers are built and physics tables
   constructed.
4. In an MT run, ``UserActionInitialization::Build`` is called for each worker
   thread and the thread-local construction is repeated.

During each run:

1. ``BeginOfRunAction`` is called on the main thread, and then on
   all worker threads in parallel (in MT mode).
2. The user's primary generator fills in the initial primary vertices for the
   event. Then ``BeginOfEventAction`` is called, the tracking and then stepping
   actions are called beneath that, and finally ``EndOfEventAction`` is called.
   This is repeated independently on each thread until the number of requested
   events has completed.
3. ``EndOfRunAction`` is called on each worker thread as soon as no more events
   are left. After all threads are complete (in MT mode) the main thread calls
   ``EndOfRunAction`` as well.

Running with Celeritas
----------------------

The ordering in the previous section determines how Celeritas is integrated
into Geant4 applications.

- The decision to disable Celeritas, and the particles to offload, must be
  handled *before* adding tracking managers.
- User macros and code must be able to override Celeritas setup options
  *before* the ``Initialize`` call.
- Celeritas shared parameters must be initialized *after* the geometry and
  physics tables are available and initialized through Geant4.
- Celeritas setup options must be defined in a long-lived data structure,
  pointed to by the :cpp:class:`celeritas::SetupOptionsMessenger`, which can
  also be modified by C++ code before construction.
- If run in an MPI-aware application, the first call to the Celeritas loggers
  must be *after* MPI is initialized (either via Celeritas'
  :cpp:class:`celeritas::ScopedMpiInit` or directly through ``MPI_Init``).

Additionally, Celeritas "local" data structures must be initialized *on the
Geant4 thread* that uses them. This is necessary only because hit
reconstruction and other Geant4 integration components use Geant4 thread-local
data structures (and, implicitly, some thread-local Geant4 allocators).

For the high-level integration classes, this means:

- MPI and logging are initialized at the first call to the integration
  singletons, which can be before or after the run manager is
  constructed.
- Code-provided options must be "moved" into the integration class before run
  initialization. This allows the setup messenger options to update the values
  after assignment in the code, and it informs the developer that subsequent
  programmatic changes to the options are prohibited.
- The decision to disable Celeritas based on user environment settings is made
  during physics initialization when using the tracking manager.
- Shared and thread-local Celeritas data structures are created during
  ``BeginOfRunAction``.

.. toctree::
   :maxdepth: 2

   geant4-interface/high-level.rst
   geant4-interface/setup.rst
   geant4-interface/details.rst
   geant4-interface/low-level.rst
