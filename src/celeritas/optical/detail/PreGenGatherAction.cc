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

#include "PreGenGatherExecutor.hh"
#include "PreGenParams.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, optical properties, and storage.
 */
PreGenGatherAction::PreGenGatherAction(ActionId id, AuxId data_id)
    : id_(id), data_id_(data_id)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(data_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view PreGenGatherAction::description() const
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
    auto& optical_state
        = get<OpticalGenState<MemSpace::native>>(state.aux(), data_id_);

    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenGatherExecutor{optical_state.store.ref()});
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
}  // namespace optical
}  // namespace celeritas
