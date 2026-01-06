//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/CelerRun.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <DDG4/Geant4Action.h>
#include <DDG4/Geant4RunAction.h>

namespace celeritas
{
namespace dd
{
//---------------------------------------------------------------------------//
/*!
 * DDG4 action plugin for Celeritas run action.
 */
class CelerRun final : public dd4hep::sim::Geant4RunAction
{
  public:
    // Standard constructor
    CelerRun(dd4hep::sim::Geant4Context* ctxt, std::string const& name);

    // Run action callbacks
    void begin(G4Run const* run) final;
    void end(G4Run const* run) final;

  protected:
    // Define standard assignments and constructors
    DDG4_DEFINE_ACTION_CONSTRUCTORS(CelerRun);
    ~CelerRun() final;
};

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas
