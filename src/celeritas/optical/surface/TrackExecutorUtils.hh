//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/TrackExecutorUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/TrackSlotExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Predicate on whether the track is undergoing a boundary interaction.
 */
template<class F>
struct IsBoundaryAction
{
    ActionId action;
    F const& select_boundary_action;

    inline CELER_FUNCTION bool operator()(CoreTrackView const& c) const
    {
        return false && action == select_boundary_action(c);
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_boundary_roughness_executor(CoreParamsPtr<MemSpace::native> params,
                                 CoreStatePtr<MemSpace::native> state,
                                 ActionId action,
                                 T&& apply_track)
{
    CELER_EXPECT(action);
    return ConditionalTrackSlotExecutor{
        params,
        state,
        IsBoundaryAction{action,
                         [](CoreTrackView const& c) {
                             return c.surface().roughness_action_id();
                         }},
        celeritas::forward<T>(apply_track)};
}

template<class T>
inline CELER_FUNCTION decltype(auto)
make_boundary_reflectivity_executor(CoreParamsPtr<MemSpace::native> params,
                                    CoreStatePtr<MemSpace::native> state,
                                    ActionId action,
                                    T&& apply_track)
{
    CELER_EXPECT(action);
    return ConditionalTrackSlotExecutor{
        params,
        state,
        IsBoundaryAction{action,
                         [](CoreTrackView const& c) {
                             return c.surface().reflectivity_action_id();
                         }},
        celeritas::forward<T>(apply_track)};
}

template<class T>
inline CELER_FUNCTION decltype(auto)
make_boundary_interaction_executor(CoreParamsPtr<MemSpace::native> params,
                                   CoreStatePtr<MemSpace::native> state,
                                   ActionId action,
                                   T&& apply_track)
{
    CELER_EXPECT(action);
    return ConditionalTrackSlotExecutor{
        params,
        state,
        IsBoundaryAction{action,
                         [](CoreTrackView const& c) {
                             return c.surface().interaction_action_id();
                         }},
        celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
