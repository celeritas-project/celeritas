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
 */
class CherenkovGenOffload : public G4Cerenkov
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    // Create a generator distribution for the given track and step
    G4VParticleChange*
    PostStepDoIt(G4Track const& aTrack, G4Step const& aStep) override;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
