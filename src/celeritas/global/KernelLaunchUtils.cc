//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/KernelLaunchUtils.cc
//---------------------------------------------------------------------------//
#include "KernelLaunchUtils.hh"

#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

KernelLaunchParams
compute_launch_params(ActionId action,
                      CoreParams const& params,
                      CoreState<MemSpace::device> const& state,
                      TrackOrder expected)
{
    KernelLaunchParams kernel_params;

    if (params.init()->host_ref().track_order == expected)
    {
        auto action_range = state.get_action_range(action);
        kernel_params.num_threads = celeritas::ceil_to_multiple(
            action_range.size(),
            size_type{celeritas::device().default_block_size()});
        kernel_params.threads_offset = action_range.front();
    }
    else
    {
        kernel_params.num_threads = state.size();
        kernel_params.threads_offset = ThreadId{0};
    }
    return kernel_params;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
