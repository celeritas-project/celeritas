//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cu
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "CounterExecutors.hh"

#include "detail/InitTracksExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch (device) kernel to initialize tracks and to update the corresponding
 * counters.
 */
void InitializeTracksAction::step_impl(CoreParams const& params,
                                       CoreStateDevice& state,
                                       size_type num_new_tracks) const
{
    detail::InitTracksExecutor execute{params.ptr<MemSpace::native>(),
                                       state.ptr()};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(num_new_tracks, state.stream_id(), execute);
}

//---------------------------------------------------------------------------//
/*!
 * Launch (device) kernel to update the corresponding counters.
 *
 */
void InitializeTracksAction::update_num_active(CoreParams const& params,
                                               CoreStateDevice& state) const
{
    // Store number of active tracks at the start of the loop, and update the
    // number of vacancies and initializers if num_new_tracks > 0
    {
        UpdateNumActiveExecutor execute_thread{
            state.ref().init.counters.data(), state.size()};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "update-active");
        launch_kernel(1, state.stream_id(), execute_thread);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
