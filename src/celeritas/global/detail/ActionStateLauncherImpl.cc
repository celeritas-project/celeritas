//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file src/celeritas/global/detail/ActionStateLauncherImpl.cc
//---------------------------------------------------------------------------//
#include "ActionStateLauncherImpl.hh"

#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

Range<ThreadId> compute_launch_params(ActionId action,
                                      CoreParams const& params,
                                      CoreState<MemSpace::device> const& state,
                                      TrackOrder expected)
{
    if (params.init()->host_ref().track_order == expected)
    {
        auto action_range = state.get_action_range(action);
        return range(
            action_range.front(),
            action_range.front()
                + celeritas::ceil_to_multiple(
                    action_range.size(),
                    size_type{celeritas::device().default_block_size()}));
    }
    else
    {
        return range(ThreadId{state.size()});
    }
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
