//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/OpticalPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VPhysicsConstructor.hh>

#include "celeritas/ext/GeantOpticalPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas-supported optical physics.
 */
class OpticalPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantOpticalPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit OpticalPhysics(Options const& options);

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
