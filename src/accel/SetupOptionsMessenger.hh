//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptionsMessenger.hh
//---------------------------------------------------------------------------//
#pragma once
#include <memory>
#include <vector>
#include <G4UIcommand.hh>
#include <G4UIdirectory.hh>
#include <G4UImessenger.hh>

namespace celeritas
{
struct SetupOptions;

//---------------------------------------------------------------------------//
/*!
 * Expose setup options through the Geant4 "macro" UI interface.
 *
 * The following options are exposed in the \c /celer/ command "directory":
 *
  Command              | Description
  -------------------- | -----------------------------------------------------
  geometryFile         | Override detector geometry with a custom GDML
  outputFile           | Filename for JSON diagnostic output
  physicsOutputFile    | Filename for ROOT dump of physics data
  offloadOutputFile    | Filename for HepMC3/ROOT/JSON dump of offloaded tracks
  geometryOutputFile   | Filename for GDML export
  maxNumTracks         | Number of tracks to be transported simultaneously
  maxNumEvents         | Maximum number of events in use
  maxNumSteps          | Limit on number of step iterations before aborting
  maxInitializers      | Maximum number of track initializers
  secondaryStackFactor | At least the average number of secondaries per track
  autoFlush            | Number of tracks to buffer before offloading
  maxFieldSubsteps     | Limit on substeps in field propagator
  slotDiagnosticPrefix | Print IDs of particles in all slots (expensive)

 * The following option is exposed in the \c /celer/detector/ command
 * "directory":
 *
  Command | Description
  ------- | -----------------------------------------
  enabled | Call back to Geant4 sensitive detectors

 * If a CUDA/HIP device is available, additional options are available under \c
 * /celer/device/ (or a DEPRECATED /celer/cuda path, to be removed in v0.7):
 *
  Command        | Description
  -------------- | ------------------------------------------------
  stackSize      | Set the CUDA per-thread stack size for VecGeom
  heapSize       | Set the CUDA per-thread heap size for VecGeom
  actionTimes    | Add timers around every action (may reduce performance)
  defaultStream  | Launch all kernels on the default stream (DEPRECATED)
 *
 * \warning The given SetupOptions should be global *or* otherwise must exceed
 * the scope of this UI messenger.
 */
class SetupOptionsMessenger : public G4UImessenger
{
  public:
    // Construct with a reference to a setup options instance
    explicit SetupOptionsMessenger(SetupOptions* options);

    // Default destructor
    ~SetupOptionsMessenger() override;

  protected:
    void SetNewValue(G4UIcommand* command, G4String newValue) override;
    G4String GetCurrentValue(G4UIcommand* command) override;

  private:
    std::vector<std::unique_ptr<G4UIdirectory>> directories_;
    std::vector<std::unique_ptr<G4UIcommand>> commands_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
