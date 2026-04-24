//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ScintillationGenOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Scintillation.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A replacement for Geant4's \c G4Scintillation process which constructs \c
 * GeneratorDistributionData from a \c PostStepDoIt call.
 */
class ScintillationGenOffload : public G4Scintillation
{
  public:
    // Create a generator distribution for the given track and step
    G4VParticleChange*
    PostStepDoIt(G4Track const& aTrack, G4Step const& aStep) override;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
