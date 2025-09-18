//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/TrackSlotExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Whether the track is undergoing a surface crossing and it matches the given
 * step and model.
 */
struct IsSurfaceModelEqual
{
    SurfacePhysicsOrder step;
    SurfaceModelId model;

    //! Whether the surface model should be executed for the track
    CELER_FUNCTION bool operator()(CoreTrackView const& track) const
    {
        auto s_phys = track.surface_physics();
        return s_phys.is_crossing_boundary()
               && s_phys.surface_model(
                            s_phys.traversal_direction(track.geometry().dir()),
                            step)
                          .surface_model()
                      == model;
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a track slot executor that for a given surface step and model.
 *
 * The executor will launch kernels only on tracks which are undergoing a
 * boundary crossing.
 */
template<class T>
inline CELER_FUNCTION decltype(auto)
make_surface_physics_executor(CoreParamsPtr<MemSpace::native> params,
                              CoreStatePtr<MemSpace::native> const& state,
                              SurfacePhysicsOrder step,
                              SurfaceModelId model,
                              T&& apply_track)
{
    CELER_EXPECT(step != SurfacePhysicsOrder::size_);
    CELER_EXPECT(model);
    return ConditionalTrackSlotExecutor{params,
                                        state,
                                        IsSurfaceModelEqual{step, model},
                                        celeritas::forward<T>(apply_track)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
