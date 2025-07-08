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
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Make a track slot executor for roughness models during a boundary crossing.
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
        [action](CoreTrackView const& c) {
            return c.is_crossing_boundary()
                   && action == c.surface().roughness_action_id();
        },
        celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
/*!
 * Make a track slot executor for reflectivity models during a boundary
 * crossing.
 */
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
        [action](CoreTrackView const& c) {
            return c.is_crossing_boundary()
                   && action == c.surface().reflectivity_action_id();
        },
        celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
/*!
 * Make a track slot executor for interaction models during a boundary
 * crossing.
 */
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
        [action](CoreTrackView const& c) {
            return c.is_crossing_boundary()
                   && action == c.surface().interaction_action_id();
        },
        celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
