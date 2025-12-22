//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>
#include <DDG4/Geant4Action.h>
#include <DDG4/Geant4PhysicsList.h>
#include <G4VModularPhysicsList.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * DDG4 action plugin for Celeritas tracking manager integration (TMI).
 */
class DDcelerTMI final : public dd4hep::sim::Geant4PhysicsList
{
  public:
    // Standard constructor
    DDcelerTMI(dd4hep::sim::Geant4Context* ctxt, std::string const& name);

    // Delete copy/move
    DDG4_DEFINE_ACTION_CONSTRUCTORS(DDcelerTMI);

    // constructPhysics callback
    virtual void constructPhysics(G4VModularPhysicsList* physics) final;

  private:
    int max_num_tracks_{0};
    int init_capacity_{0};
    std::vector<std::string> ignore_processes_;

    // Make options for Celeritas tracking manager
    SetupOptions make_options();
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
