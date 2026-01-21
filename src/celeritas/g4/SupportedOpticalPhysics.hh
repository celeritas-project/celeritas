//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SupportedOpticalPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VPhysicsConstructor.hh>

#include "celeritas/ext/GeantOpticalPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas-supported optical physics processes.
 */
class SupportedOpticalPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantOpticalPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit SupportedOpticalPhysics(Options const& options);

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    Options options_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
