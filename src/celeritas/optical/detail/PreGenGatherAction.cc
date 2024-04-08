//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenGatherAction.cc
//---------------------------------------------------------------------------//
#include "PreGenGatherAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "OpticalGenStorage.hh"
#include "PreGenGatherExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, optical properties, and storage.
 */
PreGenGatherAction::PreGenGatherAction(ActionId id, SPGenStorage storage)
    : id_(id), storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string PreGenGatherAction::description() const
{
    return "gather pre-step data to generate optical distributions";
}

//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data.
 */
void PreGenGatherAction::execute(CoreParams const& params,
                                 CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenGatherExecutor{storage_->obj.state<MemSpace::native>(
            state.stream_id(), state.size())});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void PreGenGatherAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
