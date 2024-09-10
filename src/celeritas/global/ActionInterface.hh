//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

//! Interface called at end of the core stepping loop
using CoreEndRunGatherActionInterface
    = EndRunGatherActionInterface<CoreParams, CoreState>;

// TODO: Remove in v0.6
using ActionOrder [[deprecated]] = StepActionOrder;

// TODO: Remove in v0.6
class [[deprecated]] ExplicitCoreActionInterface
    : public CoreStepActionInterface
{
    //! Execute the action with host data
    void step(CoreParams const& params, CoreStateHost& state) const final
    {
        return this->execute(params, state);
    }

    //! Execute the action with device data
    void step(CoreParams const& params, CoreStateDevice& state) const final
    {
        return this->execute(params, state);
    }

    //! Execute the action with host data
    virtual void execute(CoreParams const&, CoreStateHost&) const = 0;

    //! Execute the action with device data
    virtual void execute(CoreParams const&, CoreStateDevice&) const = 0;
};

//---------------------------------------------------------------------------//
// HELPER STRUCTS
//---------------------------------------------------------------------------//
//! Action order/ID tuple for comparison in sorting
struct OrderedAction
{
    StepActionOrder order;
    ActionId id;

    //! Ordering comparison for an action/ID
    CELER_CONSTEXPR_FUNCTION bool operator<(OrderedAction const& other) const
    {
        if (this->order < other.order)
            return true;
        if (this->order > other.order)
            return false;
        return this->id < other.id;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Checks that the TrackOrder will sort tracks by actions applied at the given
 * StepActionOrder.
 *
 * This should match the mapping in the \c SortTracksAction constructor.
 *
 * \todo Have a single source of truth for mapping TrackOrder to
 * StepActionOrder.
 */
inline bool is_action_sorted(StepActionOrder action, TrackOrder track)
{
    return (action == StepActionOrder::post
            && track == TrackOrder::sort_step_limit_action)
           || (action == StepActionOrder::along
               && track == TrackOrder::sort_along_step_action)
           || (track == TrackOrder::sort_action
               && (action == StepActionOrder::post
                   || action == StepActionOrder::along));
}

//---------------------------------------------------------------------------//
/*!
 * Whether track sorting is enabled.
 */
inline constexpr bool is_action_sorted(TrackOrder track)
{
    return static_cast<int>(track) > static_cast<int>(TrackOrder::shuffled);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
