//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HandBackTestAction.cc
//---------------------------------------------------------------------------//
#include "HandBackTestAction.hh"

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID and the post-step action that triggers it.
 */
HandBackTestAction::HandBackTestAction(ActionId id, ActionId trigger)
    : id_{id}, trigger_{trigger}
{
    CELER_EXPECT(id_);
    CELER_EXPECT(trigger_);
}

//---------------------------------------------------------------------------//
/*!
 * Mark tracks on host.
 */
void HandBackTestAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    auto execute = make_active_track_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              HandBackTestExecutor{trigger_});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
