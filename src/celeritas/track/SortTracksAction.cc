//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SortTracksAction.cc
//---------------------------------------------------------------------------//
#include "SortTracksAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/track/detail/TrackSortUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Short name for the action
 */
std::string SortTracksAction::label() const
{
    switch (action_order_)
    {
        case ActionOrder::sort_start:
            return "sort-tracks-start";
        default:
            return "sort-tracks";
    }
}
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void SortTracksAction::execute(ParamsHostCRef const&,
                               StateHostRef& states) const
{
    switch (track_order_)
    {
        case TrackOrder::partition_status:
            detail::partition_tracks_by_status(states);
            break;
        case TrackOrder::sort_step_limit_action:
            detail::sort_tracks_by_action_id(states);
            break;
        default:
            break;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void SortTracksAction::execute(ParamsDeviceCRef const&,
                               StateDeviceRef& states) const
{
    switch (track_order_)
    {
        case TrackOrder::partition_status:
            detail::partition_tracks_by_status(states);
            break;
        case TrackOrder::sort_step_limit_action:
            detail::sort_tracks_by_action_id(states);
            break;
        default:
            break;
    }
}

}  // namespace celeritas
