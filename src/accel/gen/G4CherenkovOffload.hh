//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/gen/G4CherenkovOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Cerenkov.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A replacement for Geant4's \c G4Cerenkov process which constructs \c
 * GeneratorDistributionData from a \c PostStepDoIt call.
 *
 * This process should have stacking photons set to false so that photons are
 * not initialized in Geant4.
 */
class G4CherenkovOffload : public G4Cerenkov
{
  public:
    // Create a generator distribution for the given track and step
    G4VParticleChange*
    PostStepDoIt(G4Track const& aTrack, G4Step const& aStep) override;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
