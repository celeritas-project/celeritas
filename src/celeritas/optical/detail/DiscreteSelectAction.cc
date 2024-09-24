//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/DiscreteSelectAction.cc
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "DiscreteSelectExecutor.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an action ID.
 */
DiscreteSelectAction::DiscreteSelectAction(ActionId id)
    : ConcreteAction(
        id, "optical-discrete-select", "select a discrete optical interaction")
{
    CELER_EXPECT(id);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete selection action on host.
 */
void DiscreteSelectAction::step(CoreParams const& params,
                                CoreStateHost& state) const
{
    // TODO: Implement with optical action launchers
    // auto execute =
    // make_action_track_executor(params.ptr<MemSpace::native>(),
    //                                           state.ptr(),
    //                                           this->action_id(),
    //                                           DiscreteSelectExecutor{});
    // return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete selection action on device.
 */
#if !CELER_USE_DEVICE
void DiscreteSelectAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
