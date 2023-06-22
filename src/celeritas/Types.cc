//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a state of matter.
 */
char const* to_cstring(MatterState value)
{
    static EnumStringMapper<MatterState> const to_cstring_impl{
        "unspecified",
        "solid",
        "liquid",
        "gas",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an action order.
 */
char const* to_cstring(ActionOrder value)
{
    static EnumStringMapper<ActionOrder> const to_cstring_impl{
        "start",
        "sort_start",
        "pre",
        "sort_pre",
        "along",
        "sort_along",
        "pre_post",
        "sort_pre_post",
        "post",
        "post_post",
        "end",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a track ordering policy.
 */
char const* to_cstring(TrackOrder value)
{
    static EnumStringMapper<TrackOrder> const to_cstring_impl{
        "unsorted",
        "shuffled",
        "partition_status",
        "sort_along_step_action",
        "sort_step_limit_action",
        "sort_action",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Checks that the TrackOrder will sort tracks by actions applied at the given
 * ActionOrder. This should match the mapping in the \c SortTracksAction
 * constructor.
 *
 * TODO: Have a single source of truth for mapping TrackOrder to ActionOrder
 */
bool is_action_sorted(ActionOrder action, TrackOrder track)
{
    return (action == ActionOrder::post
            && track == TrackOrder::sort_step_limit_action)
           || (action == ActionOrder::along
               && track == TrackOrder::sort_along_step_action)
           || (track == TrackOrder::sort_action
               && (action == ActionOrder::post || action == ActionOrder::along));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
