//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/TrackSlotExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

struct IsSurfaceModelEqual
{
    SurfacePhysicsOrder step;
    SurfaceModelId model;

    inline CELER_FUNCTION bool operator()(CoreTrackView const& track) const
    {
        if (!track.surface_physics().is_crossing_boundary())
        {
            return false;
        }
        // #if !CELER_DEVICE_COMPILE
        //         if (!is_soft_unit_vector(track.geometry().dir()))
        //         {
        //             CELER_LOG_LOCAL(error)
        //                 << " track surface: "
        //                 << track.surface_physics().surface().unchecked_get()
        //                 << " (invalid: " << SurfaceId{}.unchecked_get()
        //                 << ", is_crossing_boundary: "
        //                 << track.surface_physics().is_crossing_boundary() <<
        //                 ")";
        //         }
        // #endif
        return track.surface_model(step).surface_model() == model;
    }
};

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
