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

struct IsSurfaceModelEqual
{
    SurfacePhysicsOrder step;
    SurfaceModelId model;

    inline CELER_FUNCTION bool operator()(CoreTrackView const& track) const
    {
        return track.surface_physics().is_crossing_boundary()
               && track.surface_model(step).surface_model() == model;
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
