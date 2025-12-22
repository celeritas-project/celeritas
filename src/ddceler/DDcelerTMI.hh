//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <CeleritasG4.hh>
#include <DDG4/Geant4Action.h>
#include <DDG4/Geant4PhysicsList.h>
#include <G4VModularPhysicsList.hh>

using Geant4Context = dd4hep::sim::Geant4Context;
using Geant4Action = dd4hep::sim::Geant4Action;
using Geant4PhysicsList = dd4hep::sim::Geant4PhysicsList;

namespace dd4hep
{
namespace sim
{
//---------------------------------------------------------------------------//
/*!
 * DDG4 action plugin for Celeritas tracking manager integration (TMI).
 */
class DDcelerTMI final : public Geant4PhysicsList
{
  public:
    // Standard constructor
    DDcelerTMI(Geant4Context* ctxt, std::string const& name);

    // Make options for Celeritas tracking manager
    celeritas::SetupOptions makeOptions();

    // constructPhysics callback
    virtual void constructPhysics(G4VModularPhysicsList* physics) final;

  protected:
    // Define standard assignments and constructors
    DDG4_DEFINE_ACTION_CONSTRUCTORS(DDcelerTMI);

    int max_num_tracks_{0};
    int init_capacity_{0};
    std::vector<std::string> ignore_processes_;
};
//---------------------------------------------------------------------------//
}  // namespace sim
}  // namespace dd4hep
