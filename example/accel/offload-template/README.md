Minimal Geant4-Celeritas offloading template
============================================

Template for Geant4 applications with Celeritas physics offloading capabilities.
It shows how to link Celeritas against Geant4 in the `CMakeLists.txt` and the
Geant4 classes needed to initialize Celeritas, offload events, and recover step
information.

# Dependencies
- Geant4 v10.7 or newer
- Celeritas v0.5 or newer
  - `CELERITAS_USE_Geant4=ON`
  - `CELERITAS_USE_VecGeom=ON`

# Build and run
```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./main
```
Correct execution is verified by Geant4 and Celeritas log messages.

# Boilerplate offloading code

- `Celeritas.[hh/cc]`: Defines the needed components for a Celeritas offload
  execution:
  - **Setup options** (memory allocation, physics, field, scoring, and so on)
  - **Shared parameters** used in the run (materials, physics processes,
    cross-section tables, and so on)
  - **Transporter** (execute/manage the particle transport)
- `G4UserRunAction`
  - `BeginOfRunAction`: Initialize Celeritas global shared data on master and
    worker threads
  - `EndOfRunAction`: Clear data and return Celeritas to an invalid state
- `G4UserEventAction`
  - `BeginOfEventAction`: Initialize event in Celeritas
  - `EndOfEventAction`: Flush remaining particles
- `G4UserTrackingAction`
  - `PreUserTrackingAction`: Kill particles in Geant4; offload their state to
    Celeritas
- `G4VSensitiveDetector`
  - `ProcessHits`: Currently the *only* Celeritas callback interface to Geant4;
  at each step, Celeritas sends data back as a `G4Step` to be processed by
  Geant4
