//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadGatherAction.cc
//---------------------------------------------------------------------------//
#include "OffloadGatherAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "OffloadGatherExecutor.hh"
#include "../OffloadParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, optical properties, and storage.
 */
OffloadGatherAction::OffloadGatherAction(ActionId id, AuxId data_id)
    : id_(id), data_id_(data_id)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(data_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view OffloadGatherAction::description() const
{
    return "gather pre-step data to generate optical distributions";
}

//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data.
 */
void OffloadGatherAction::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    auto& optical_state
        = get<OpticalOffloadState<MemSpace::native>>(state.aux(), data_id_);

    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::OffloadGatherExecutor{optical_state.store.ref()});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void OffloadGatherAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
