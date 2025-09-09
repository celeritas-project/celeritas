//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ActionInterface.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
template<MemSpace M>
class CoreState;

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//
//! Interface called at beginning of the core stepping loop
using CoreBeginRunActionInterface
    = BeginRunActionInterface<CoreParams, CoreState>;

//! Action interface for core stepping loop
using CoreStepActionInterface = StepActionInterface<CoreParams, CoreState>;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the TrackOrder will sort tracks by actions at the given step order.
 */
inline constexpr bool
is_action_sorted(StepActionOrder aorder, TrackOrder torder)
{
    // CAUTION: check that this matches \c SortTracksAction::SortTracksAction
    return (aorder == StepActionOrder::post
            && torder == TrackOrder::reindex_step_limit_action)
           || (aorder == StepActionOrder::along
               && torder == TrackOrder::reindex_along_step_action)
           || (torder == TrackOrder::reindex_both_action
               && (aorder == StepActionOrder::post
                   || aorder == StepActionOrder::along));
}

//---------------------------------------------------------------------------//
/*!
 * Whether track sorting (reindexing) is enabled.
 */
inline constexpr bool is_action_sorted(TrackOrder torder)
{
    auto to_int = [](TrackOrder v) {
        return static_cast<std::underlying_type_t<TrackOrder>>(v);
    };
    return to_int(torder) >= to_int(TrackOrder::begin_reindex_)
           && to_int(torder) < to_int(TrackOrder::end_reindex_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
