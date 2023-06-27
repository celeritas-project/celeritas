//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SortTracksAction.cc
//---------------------------------------------------------------------------//
#include "SortTracksAction.hh"

#include <algorithm>
#include <iterator>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/TrackSortUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Checks whether the TrackOrder defines a sorting strategy.
 */
bool is_sort_trackorder(TrackOrder to)
{
    static TrackOrder const allowed[] = {
        TrackOrder::partition_status,
        TrackOrder::sort_step_limit_action,
        TrackOrder::sort_along_step_action,
        TrackOrder::sort_action,
    };
    return std::find(std::begin(allowed), std::end(allowed), to)
           != std::end(allowed);
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with sort ordering and track order policy.
 */
SortTracksAction::SortTracksAction(ActionId id, TrackOrder track_order)
    : id_(id), track_order_(track_order)
{
    CELER_EXPECT(id_);
    CELER_VALIDATE(is_sort_trackorder(track_order_),
                   << "track ordering policy '" << to_cstring(track_order)
                   << "' should not sort tracks");
    CELER_EXPECT(track_order != TrackOrder::sort_action);
    action_order_ = [track_order = track_order_] {
        switch (track_order)
        {
            case TrackOrder::partition_status:
                // Sort *after* setting status
                return ActionOrder::sort_start;
            case TrackOrder::sort_along_step_action:
                // Sort *before* along-step action, i.e. *after* pre-step
                return ActionOrder::sort_pre;
            case TrackOrder::sort_step_limit_action:
                // Sort *before* post-step action, i.e. *after* pre-post and
                // along-step
                return ActionOrder::sort_pre_post;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }();
}

//---------------------------------------------------------------------------//
/*!
 * Short name for the action
 */
std::string SortTracksAction::label() const
{
    switch (track_order_)
    {
        case TrackOrder::partition_status:
            return "sort-tracks-partition-status";
        case TrackOrder::sort_along_step_action:
            return "sort-tracks-along-step";
        case TrackOrder::sort_step_limit_action:
            return "sort-tracks-post-step";
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void SortTracksAction::execute(CoreParams const&, CoreStateHost& state) const
{
    detail::sort_tracks(state.ref(), track_order_);
    detail::count_tracks_per_action(
        state.ref(),
        state.action_thread_offsets()[AllItems<ThreadId, MemSpace::host>{}],
        state.action_thread_offsets(),
        track_order_);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void SortTracksAction::execute(CoreParams const&, CoreStateDevice& state) const
{
    detail::sort_tracks(state.ref(), track_order_);
    detail::count_tracks_per_action(
        state.ref(),
        state.native_action_thread_offsets()[AllItems<ThreadId,
                                                      MemSpace::device>{}],
        state.action_thread_offsets(),
        track_order_);
}

//---------------------------------------------------------------------------//
/*!
 * Set host data at the beginning of a run
 */
void SortTracksAction::begin_run(CoreParams const& params, CoreStateHost& state)
{
    state.num_actions(params.action_reg()->num_actions() + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Set device data at the beginning of a run
 */
void SortTracksAction::begin_run(CoreParams const& params,
                                 CoreStateDevice& state)
{
    state.num_actions(params.action_reg()->num_actions() + 1);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
