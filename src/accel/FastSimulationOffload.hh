//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VFastSimulationModel.hh>
#include <G4Version.hh>

namespace celeritas
{
class SharedParams;
class LocalTransporter;

//---------------------------------------------------------------------------//
/*!
 * Offload tracks to Celeritas via G4VFastSimulationModel interface
 *
 * This class must be constructed locally on each worker thread/task, typically
 * within the application's concrete implementation of
 * `G4VUserDetectorConstruction::ConstructSDandField()`.
 */
class FastSimulationOffload final : public G4VFastSimulationModel
{
  public:
    // Construct with name, shared (across threads) params, and thread-local
    // transporter
    FastSimulationOffload(G4String const& name,
                          SharedParams const* params,
                          LocalTransporter* local);

    // Construct with name, region, shared (across threads) params, and
    // thread-local transporter
    FastSimulationOffload(G4String const& name,
                          G4Envelope* region,
                          SharedParams const* params,
                          LocalTransporter* local);

    // Return true if model is applicable to the `G4ParticleDefinition`
    G4bool IsApplicable(G4ParticleDefinition const& particle) override;

    // Return true if model is applicable to dynamic state of `G4FastTrack`
    G4bool ModelTrigger(G4FastTrack const& track) override;

    // Apply model
    void DoIt(G4FastTrack const& track, G4FastStep& step) override;

#if G4VERSION_NUMBER > 1110
    //! Complete processing of buffered tracks
    void Flush() override;
#endif

  private:
    SharedParams const* params_{nullptr};
    LocalTransporter* transport_{nullptr};
};
}  // namespace celeritas
