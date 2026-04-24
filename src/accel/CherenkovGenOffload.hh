//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CherenkovGenOffload.hh
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
 * Allows offloading directly to \c LocalOpticalGenOffload by skipping the
 * secondary photon initialization and instead pushing a \c
 * GeneratorDistributionData for the given step.
 */
class CherenkovGenOffload : public G4Cerenkov
{
  public:
    // Create a generator distribution for the given track and step
    G4VParticleChange*
    PostStepDoIt(G4Track const& aTrack, G4Step const& aStep) override;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
