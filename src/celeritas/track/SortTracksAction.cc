//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SortTracksAction.cc
//---------------------------------------------------------------------------//
#include "SortTracksAction.hh"

#include <algorithm>
#include <iterator>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/TrackSortUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Checks whether the TrackOrder sort tracks using an ActionId.
 */
bool is_sort_by_action(TrackOrder torder)
{
    auto to_int = [](TrackOrder v) {
        return static_cast<std::underlying_type_t<TrackOrder>>(v);
    };
    return to_int(torder) >= to_int(TrackOrder::begin_reindex_action_)
           && to_int(torder) < to_int(TrackOrder::end_reindex_action_);
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
    CELER_VALIDATE(is_action_sorted(track_order_),
                   << "track ordering policy '" << track_order
                   << "' should not sort tracks");
    CELER_EXPECT(track_order != TrackOrder::reindex_both_action);

    // CAUTION: check that this matches \c is_action_sorted
    action_order_ = [track_order] {
        switch (track_order)
        {
            case TrackOrder::reindex_status:
                // Partition *after* setting status
                return StepActionOrder::sort_start;
            case TrackOrder::reindex_along_step_action:
                // Sort *before* along-step action, i.e. *after* pre-step
                return StepActionOrder::sort_pre;
            case TrackOrder::reindex_step_limit_action:
                // Sort *before* post-step action, i.e. *after* pre-post and
                // along-step
                return StepActionOrder::sort_pre_post;
            case TrackOrder::reindex_particle_type:
                // Sort at the beginning of the step
                return StepActionOrder::sort_start;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }();
}

//---------------------------------------------------------------------------//
/*!
 * Short name for the action.
 */
std::string_view SortTracksAction::label() const
{
    switch (track_order_)
    {
        case TrackOrder::reindex_status:
            return "sort-tracks-status";
        case TrackOrder::reindex_along_step_action:
            return "sort-tracks-along-step";
        case TrackOrder::reindex_step_limit_action:
            return "sort-tracks-post-step";
        case TrackOrder::reindex_particle_type:
            return "sort-tracks-start";
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void SortTracksAction::step(CoreParams const&, CoreStateHost& state) const
{
    detail::sort_tracks(state.ref(), track_order_);
    if (is_sort_by_action(track_order_))
    {
        detail::count_tracks_per_action(
            state.ref(),
            state.action_thread_offsets()[AllItems<ThreadId, MemSpace::host>{}],
            state.action_thread_offsets(),
            track_order_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void SortTracksAction::step(CoreParams const&, CoreStateDevice& state) const
{
    detail::sort_tracks(state.ref(), track_order_);
    if (is_sort_by_action(track_order_))
    {
        detail::count_tracks_per_action(
            state.ref(),
            state.native_action_thread_offsets()[AllItems<ThreadId,
                                                          MemSpace::device>{}],
            state.action_thread_offsets(),
            track_order_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Set host data at the beginning of a run
 */
void SortTracksAction::begin_run(CoreParams const& params, CoreStateHost& state)
{
    CELER_VALIDATE(state.action_thread_offsets().size()
                       == params.action_reg()->num_actions() + 1,
                   << "state action size is incorrect: actions might have "
                      "been added after creating states");
}

//---------------------------------------------------------------------------//
/*!
 * Set device data at the beginning of a run
 */
void SortTracksAction::begin_run(CoreParams const& params,
                                 CoreStateDevice& state)
{
    CELER_VALIDATE(state.action_thread_offsets().size()
                       == params.action_reg()->num_actions() + 1,
                   << "state action size is incorrect: actions might have "
                      "been added after creating states");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
