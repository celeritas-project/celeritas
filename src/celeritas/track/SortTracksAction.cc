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
        case ActionOrder::start:
            return "sort-tracks-start";
        default:
            return "sort-tracks";
    }
}
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void SortTracksAction::execute(CoreHostRef const& core) const
{
    if (track_order_ == TrackOrder::partition_status)
    {
        detail::partition_tracks_by_status(core.states);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void SortTracksAction::execute(CoreDeviceRef const& core) const
{
    if (track_order_ == TrackOrder::partition_status)
    {
        detail::partition_tracks_by_status(core.states);
    }
}

}  // namespace celeritas
