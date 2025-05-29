//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CeleritasG4.hh>
#include <DDG4/Geant4Action.h>
#include <DDG4/Geant4PhysicsList.h>

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
class DDcelerTMI : public Geant4PhysicsList
{
  protected:
    // Define standard assignments and constructors
    DDG4_DEFINE_ACTION_CONSTRUCTORS(DDcelerTMI);

  public:
    // Standard constructor
    inline DDcelerTMI(Geant4Context* ctxt, std::string const& nam);

    // Default destructor
    virtual ~DDcelerTMI() = default;

    // Make options for Celeritas tracking manager
    celeritas::SetupOptions makeOptions();

    // constructPhysics callback
    virtual void constructPhysics(G4VModularPhysicsList* physics) override;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Standard constructor
 */
DDcelerTMI::DDcelerTMI(Geant4Context* ctxt, std::string const& nam)
    : Geant4PhysicsList(ctxt, nam)
{
}

//---------------------------------------------------------------------------//
}  // namespace sim
}  // namespace dd4hep
