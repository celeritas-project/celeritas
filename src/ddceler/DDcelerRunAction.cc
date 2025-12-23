//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerRunAction.cc
//---------------------------------------------------------------------------//
#include "DDcelerRunAction.hh"

#include <DD4hep/InstanceCount.h>
#include <DDG4/Factories.h>

#include "accel/TrackingManagerIntegration.hh"

using TMI = celeritas::TrackingManagerIntegration;
using Geant4Context = dd4hep::sim::Geant4Context;
using Geant4RunAction = dd4hep::sim::Geant4RunAction;
using InstanceCount = dd4hep::InstanceCount;

namespace celeritas
{
namespace ddceler
{
//---------------------------------------------------------------------------//
/*!
 * Standard constructor.
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
}  // namespace ddceler
}  // namespace celeritas

DECLARE_GEANT4ACTION_NS(celeritas::ddceler, DDcelerRunAction)
