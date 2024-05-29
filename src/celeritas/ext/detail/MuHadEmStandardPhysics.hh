//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/MuHadEmStandardPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VPhysicsConstructor.hh>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct EM standard physics not implemented in Celeritas.
 */
class MuHadEmStandardPhysics : public G4VPhysicsConstructor
{
  public:
    // Set up during construction
    explicit MuHadEmStandardPhysics(int verbosity);

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    void construct_particle();
    void construct_process();
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
