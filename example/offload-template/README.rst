Minimal Geant4-Celeritas offloading template
============================================

Template for Geant4 applications with Celeritas physics offloading capabilities.
It shows how to link Celeritas against Geant4 in the `CMakeLists.txt` and the
Geant4 classes needed to initialize Celeritas, offload events, and recover step
information.

# Dependencies
- Geant4 v11 or newer
- Celeritas v0.5 or newer
  - `CELERITAS_USE_Geant4=ON`

# Build and run
```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
$ export CELER_DISABLE_PARALLEL=1
$ ./main
```

# Boilerplate offloading code
- `Celeritas.[hh/cc]`: Defines the needed components for a Celeritas offload
  execution:
  - **Setup options** (memory allocation, physics, field, scoring, and so on)
  - **Shared parameters** used in the run (materials, physics processes,
    cross-section tables, and so on)
  - **Transporter** (execute/manage the particle transport)
  - **Simple Offload** simplified user-interface. Each `SimpleOffload` call is
    briefly described below.
- `G4VUserActionInitialization`
  - `Build`: Construct Celeritas Simple Offload interface with user-defined
    options (from `Celeritas.cc`) and assign the Celeritas tracking manager to the appropriate particles
- `G4UserRunAction`
  - `BeginOfRunAction`: Initialize Celeritas global shared data on master and
    worker threads
  - `EndOfRunAction`: Clear data and return Celeritas to an invalid state
- `G4UserEventAction`
  - `BeginOfEventAction`: Initialize event in Celeritas
  - `EndOfEventAction`: Flush remaining particles
- `G4VSensitiveDetector`
  - `ProcessHits`: Currently the *only* Celeritas callback interface to Geant4;
  at each step, Celeritas sends data back as a `G4Step` to be processed by
  Geant4
