//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/DiscreteSelectAction.cu
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
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
 * Launch the discrete selection action on device.
 */
void DiscreteSelectAction::step(CoreParams const& params,
                                CoreStateDevice& state) const
{
    // TODO: Implement with optical action launchers
    // auto execute =
    // make_action_track_executor(params.ptr<MemSpace::native>(),
    //                                           state.ptr(),
    //                                           this->action_id(),
    //                                           DiscreteSelectExecutor{});
    // static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    // launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
