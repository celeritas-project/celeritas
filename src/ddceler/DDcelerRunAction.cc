//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerRunAction.cc
//---------------------------------------------------------------------------//
#include "DDcelerRunAction.hh"

#include <CeleritasG4.hh>
#include <DDG4/Factories.h>

using TMI = celeritas::TrackingManagerIntegration;

namespace dd4hep
{
namespace sim
{
//---------------------------------------------------------------------------//
/*!
 * Standard constructor
 */
DDcelerRunAction::DDcelerRunAction(Geant4Context* ctxt, std::string const& name)
    : Geant4RunAction(ctxt, name)
{
    InstanceCount::increment(this);
}

//---------------------------------------------------------------------------//

DDcelerRunAction::~DDcelerRunAction()
{
    InstanceCount::decrement(this);
}

//---------------------------------------------------------------------------//

void DDcelerRunAction::begin(G4Run const* run)
{
    TMI::Instance().BeginOfRunAction(run);
}

//---------------------------------------------------------------------------//

void DDcelerRunAction::end(G4Run const* run)
{
    TMI::Instance().EndOfRunAction(run);
}

//---------------------------------------------------------------------------//
}  // namespace sim
}  // namespace dd4hep

DECLARE_GEANT4ACTION(DDcelerRunAction)
