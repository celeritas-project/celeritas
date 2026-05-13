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

#include "detail/InitTracksExecutor.hh"
#include "detail/UpdateNumActiveExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch (device) kernels to initialize tracks and to update the corresponding
 * counters.
 */
void InitializeTracksAction::step_impl(CoreParams const& params,
                                       CoreStateDevice& state,
                                       size_type num_new_tracks) const
{
    {
        detail::InitTracksExecutor execute{params.ptr<MemSpace::native>(),
                                           state.ptr()};
        static ActionLauncher<decltype(execute)> const launch_kernel(*this);
        launch_kernel(num_new_tracks, state.stream_id(), execute);
    }
    {
        detail::UpdateNumActiveExecutor execute_thread{
            params.ptr<MemSpace::native>(), state.ptr()};
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            *this);
        launch_kernel(1, state.stream_id(), execute_thread);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
