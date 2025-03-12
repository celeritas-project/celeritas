//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <accel/TrackingManagerIntegration.hh>

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
RunAction::RunAction() : G4UserRunAction() {}

//---------------------------------------------------------------------------//
/*!
 * Initialize master and worker threads in Celeritas.
 */
void RunAction::BeginOfRunAction(G4Run const* run)
{
    celeritas::TrackingManagerIntegration::Instance().BeginOfRunAction(run);
}

//---------------------------------------------------------------------------//
/*!
 * Clear local data and return Celeritas to an invalid state.
 */
void RunAction::EndOfRunAction(G4Run const* run)
{
    celeritas::TrackingManagerIntegration::Instance().EndOfRunAction(run);
}

}  // namespace example
}  // namespace celeritas
