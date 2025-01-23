Geant4-Celeritas offloading template
====================================

This is a template for Geant4 applications with Celeritas physics offloading
capabilities. It shows how to link Celeritas against Geant4 in the
:code:`CMakeLists.txt` and the Geant4 classes needed to initialize Celeritas,
offload events, and recover step information.

Dependencies
------------

- Geant4 v11 or newer
- Celeritas v0.5 or newer with ``CELERITAS_USE_Geant4=ON``

Build and run
-------------

.. code-block:: sh

   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   $ export CELER_DISABLE_PARALLEL=1 # if Celeritas is built with MPI
   $ ./run-offload


Boilerplate offloading code
---------------------------

:file:`Celeritas.{hh,cc}`
   Defines the needed components for a Celeritas offload execution:

   - **Setup options**: memory allocation, physics, field, scoring, and so on
   - **Shared parameters** used in the run: materials, physics processes
     cross-section tables, and so on
   - **Transporter**: execute/manage the particle transport
   - **Simple Offload**: simplified user interface. Each ``SimpleOffload`` call
     is briefly described below.

:code:`G4VUserActionInitialization`
  :code:`Build` and `BuildForMaster` construct Celeritas Simple Offload
    interface with user-defined options (from ``Celeritas.cc``) and assign the
    Celeritas tracking manager to the appropriate particles.

:code:`G4UserRunAction`
  :code:`BeginOfRunAction` initializes Celeritas global shared data on master
  and worker threads. :code:`EndOfRunAction` clears data and finalizes
  Celeritas data.

:code:`G4UserEventAction`
  :code:`BeginOfEventAction` initializes an event in Celeritas, and
  :code:`EndOfEventAction` flushes remaining particles.

:code:`G4VSensitiveDetector`
  :code:`ProcessHits`: is currently the *only* Celeritas callback interface to
  Geant4; at each step, Celeritas sends data back as a ``G4Step`` to be
  processed by Geant4.
