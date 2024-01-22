//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPhysicsList.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ParticleDefinition.hh>
#include <G4VUserPhysicsList.hh>

#include "../GeantPhysicsOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a user-defined physics list of particles and physics processes.
 */
class GeantPhysicsList : public G4VUserPhysicsList
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Set up during construction
    explicit GeantPhysicsList(Options const& options);

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
