//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/EmStandardPhysics.hh
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
class EmStandardPhysics : public G4VPhysicsConstructor
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit EmStandardPhysics(Options const& options);

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
    // Add EM processes for muons
    void add_mu_processes(G4ParticleDefinition* p);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
