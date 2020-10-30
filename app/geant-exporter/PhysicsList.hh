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
    PhysicsList();
    ~PhysicsList();

    void ConstructParticle() override;
    void ConstructProcess() override;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
