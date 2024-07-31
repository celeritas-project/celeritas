//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OffloadGatherAction.cu
//---------------------------------------------------------------------------//
#include "OffloadGatherAction.hh"

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "OffloadGatherExecutor.hh"
#include "OffloadParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data.
 */
void OffloadGatherAction::execute(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    auto& optical_state
        = get<OpticalOffloadState<MemSpace::native>>(state.aux(), data_id_);
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::OffloadGatherExecutor{optical_state.store.ref()});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
