//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsList.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserPhysicsList.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Construct a user-defined physics list of particles and physics processes.
 */
class PhysicsList : public G4VUserPhysicsList
{
  public:
    // Construct empty
    PhysicsList();
    // Default destructor
    ~PhysicsList();

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
} // namespace geant_exporter
