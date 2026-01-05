//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/CelerRun.cc
//---------------------------------------------------------------------------//
#include "CelerRun.hh"

#include <DD4hep/InstanceCount.h>
#include <DDG4/Factories.h>

#include "accel/TrackingManagerIntegration.hh"

using TMI = celeritas::TrackingManagerIntegration;
using Geant4Context = dd4hep::sim::Geant4Context;
using Geant4RunAction = dd4hep::sim::Geant4RunAction;
using InstanceCount = dd4hep::InstanceCount;

namespace celeritas
{
namespace dd
{
//---------------------------------------------------------------------------//
/*!
 * Standard constructor.
 */
CelerRun::CelerRun(Geant4Context* ctxt, std::string const& name)
    : Geant4RunAction(ctxt, name)
{
    InstanceCount::increment(this);
}

//---------------------------------------------------------------------------//

CelerRun::~CelerRun()
{
    InstanceCount::decrement(this);
}

//---------------------------------------------------------------------------//

void CelerRun::begin(G4Run const* run)
{
    TMI::Instance().BeginOfRunAction(run);
}

//---------------------------------------------------------------------------//

void CelerRun::end(G4Run const* run)
{
    TMI::Instance().EndOfRunAction(run);
}

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas

DECLARE_GEANT4ACTION_NS(celeritas::dd, CelerRun)
