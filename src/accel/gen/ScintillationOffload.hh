//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/gen/ScintillationOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Scintillation.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A replacement for Geant4's \c G4Scintillation process which constructs \c
 * GeneratorDistributionData from a \c PostStepDoIt call.
 *
 * This process should have stacking photons set to false so that photons are
 * not initialized in Geant4.
 */
class ScintillationOffload : public G4Scintillation
{
  public:
    // Prepare physics table for particle and enforce photon stacking
    void PreparePhysicsTable(G4ParticleDefinition const&) override;

    // Create a generator distribution for the given track and step
    G4VParticleChange*
    PostStepDoIt(G4Track const& aTrack, G4Step const& aStep) override;

    // Create a generator distribution for the given track and step
    G4VParticleChange*
    AtRestDoIt(G4Track const& aTrack, G4Step const& aStep) override;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
