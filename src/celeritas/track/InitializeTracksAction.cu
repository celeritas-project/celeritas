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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to initialize tracks.
 */
void InitializeTracksAction::step_impl(CoreParams const& params,
                                       CoreStateDevice& state,
                                       size_type num_new_tracks) const
{
    detail::InitTracksExecutor execute_thread{params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              num_new_tracks,
                                              state.counters()};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(*this);
    launch_kernel(num_new_tracks, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
