//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerEmStandardPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ParticleDefinition.hh>
#include <G4VPhysicsConstructor.hh>

#include "../GeantPhysicsOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas-supported EM standard physics.
 */
class CelerEmStandardPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit CelerEmStandardPhysics(Options const& options);

    // Set up minimal EM particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    Options options_;

    // Add EM processes for photons
    void add_gamma_processes();
    // Add EM processes for electrons and positrons
    void add_e_processes(G4ParticleDefinition* p);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
