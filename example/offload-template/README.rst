====================================
Geant4-Celeritas offloading template
====================================

Template for Geant4 applications with Celeritas physics offloading capabilities.
It shows how to link Celeritas against Geant4 in the :code:`CMakeLists.txt` and
the Geant4 classes needed to initialize Celeritas, offload events, and recover
step information.

Dependencies
------------

- Geant4 v11 or newer
- Celeritas v0.5 or newer

  - :code:`CELERITAS_USE_Geant4=ON`

Build and run
-------------

.. code-block:: sh
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   $ export CELER_DISABLE_PARALLEL=1
   $ ./main

Boilerplate offloading code
---------------------------

- :code:`Celeritas.[hh/cc]`: Defines the needed components for a Celeritas
offload execution:

  - **Setup options** (memory allocation, physics, field, scoring, and so on)
  - **Shared parameters** used in the run (materials, physics processes
    cross-section tables, and so on)
  - **Transporter** (execute/manage the particle transport)
  - **Simple Offload** simplified user-interface. Each `SimpleOffload` call is
    briefly described below.

- :code:`G4VUserActionInitialization`

  - :code:`Build`: Construct Celeritas Simple Offload interface with
    user-defined options (from `Celeritas.cc`) and assign the Celeritas tracking
    manager to the appropriate particles

- :code:`G4UserRunAction`

  - :code:`BeginOfRunAction`: Initialize Celeritas global shared data on master
    and worker threads
  - :code:`EndOfRunAction`: Clear data and return Celeritas to an invalid state

- :code:`G4UserEventAction`

  - :code:`BeginOfEventAction`: Initialize event in Celeritas
  - :code:`EndOfEventAction`: Flush remaining particles

- :code:`G4VSensitiveDetector`

  - :code:`ProcessHits`: Currently the *only* Celeritas callback interface to
    Geant4; at each step, Celeritas sends data back as a `G4Step` to be
    processed by Geant4
