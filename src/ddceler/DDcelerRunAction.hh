//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerRunAction.hh
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
class DDcelerRunAction final : public Geant4RunAction
{
  public:
    // Standard constructor
    DDcelerRunAction(Geant4Context* ctxt, std::string const& name);

    // Run action callbacks
    void begin(G4Run const* run) final;
    void end(G4Run const* run) final;

  protected:
    // Define standard assignments and constructors
    DDG4_DEFINE_ACTION_CONSTRUCTORS(DDcelerRunAction);
    ~DDcelerRunAction() final;
};

//---------------------------------------------------------------------------//
}  // namespace sim
}  // namespace dd4hep
