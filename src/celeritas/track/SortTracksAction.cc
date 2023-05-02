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
#include "celeritas/global/CoreState.hh"
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
        case ActionOrder::sort_pre:
            return "sort-tracks-pre";
        default:
            return "sort-tracks";
    }
}
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data
 */
void SortTracksAction::execute(CoreParams const& params,
                               CoreStateHost& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data
 */
void SortTracksAction::execute(CoreParams const& params,
                               CoreStateDevice& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
template<MemSpace M>
void SortTracksAction::execute_impl(CoreParams const&, CoreState<M>& state) const
{
    switch (track_order_)
    {
        case TrackOrder::partition_status:
            detail::partition_tracks_by_status(state.ref());
            break;
        case TrackOrder::sort_step_limit_action:
            detail::sort_tracks_by_action_id(state.ref());
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
