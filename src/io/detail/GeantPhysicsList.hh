//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantPhysicsList.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserPhysicsList.hh>

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
    // Set up during construction
    GeantPhysicsList();

    // Set up minimal E.M. particle list
    void ConstructParticle() override;
    // Set up process list
    void ConstructProcess() override;

  private:
    // Add E.M. processes for photons
    void add_gamma_processes();
    // Add E.M. processes for electrons and positrons
    void add_e_processes();
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
