//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.cc
//---------------------------------------------------------------------------//
#include "DDcelerRunAction.hh"

#include <CeleritasG4.hh>
#include <DDG4/Factories.h>
#include <QGSP_BERT.hh>

using TMI = celeritas::TrackingManagerIntegration;

namespace dd4hep
{
namespace sim
{

//---------------------------------------------------------------------------//

void DDcelerRunAction::begin(G4Run const* run)
{
    this->info("Begin of run");
    TMI::Instance().BeginOfRunAction(run);
}

//---------------------------------------------------------------------------//

void DDcelerRunAction::end(G4Run const* run)
{
    this->info("End of run");
    TMI::Instance().EndOfRunAction(run);
}
//---------------------------------------------------------------------------//

}  // namespace sim
}  // namespace dd4hep

DECLARE_GEANT4ACTION(DDcelerRunAction)
