//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SupportedDecayPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VPhysicsConstructor.hh>

#include "celeritas/ext/GeantPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas-supported decay processes.
 *
 * This adds \c G4Decay only to particles whose decays Celeritas supports:
 * currently \f$ \mu^\pm \f$ (through \c G4MuonDecayChannel ), enabled with
 * the muon \c decay option. Unlike \c G4DecayPhysics, it does not add decay to
 * other unstable particles. Do not combine it with \c G4DecayPhysics, which
 * already adds decay to these particles.
 */
class SupportedDecayPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit SupportedDecayPhysics(Options const& options);

    // Set up decaying particles and their products
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
