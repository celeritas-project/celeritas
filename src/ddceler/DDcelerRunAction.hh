//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.hh
//---------------------------------------------------------------------------//
#pragma once

#include <DD4hep/InstanceCount.h>
#include <DDG4/Geant4Action.h>
#include <DDG4/Geant4RunAction.h>

using Geant4Context = dd4hep::sim::Geant4Context;
using Geant4Action = dd4hep::sim::Geant4Action;
using Geant4RunAction = dd4hep::sim::Geant4RunAction;

namespace dd4hep
{
namespace sim
{
//---------------------------------------------------------------------------//
/*!
 * DDG4 action plugin for Celeritas tracking manager integration (TMI).
 */
class DDcelerRunAction : public Geant4RunAction
{
  protected:
    // Define standard assignments and constructors
    DDG4_DEFINE_ACTION_CONSTRUCTORS(DDcelerRunAction);

  public:
    // Standard constructor
    inline DDcelerRunAction(Geant4Context* ctxt, std::string const& nam);

    // Default destructor
    ~DDcelerRunAction();

    // Make options for Celeritas tracking manager
    void begin(G4Run const* run) override;
    void end(G4Run const* run) override;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Standard constructor
 */
DDcelerRunAction::DDcelerRunAction(Geant4Context* ctxt, std::string const& nam)
    : Geant4RunAction(ctxt, nam)
{
    InstanceCount::increment(this);
    this->info("Constructed Geant4RunAction");
}

DDcelerRunAction::~DDcelerRunAction()
{
    this->info("Destructing Geant4RunAction");
    InstanceCount::decrement(this);
}

//---------------------------------------------------------------------------//
}  // namespace sim
}  // namespace dd4hep
