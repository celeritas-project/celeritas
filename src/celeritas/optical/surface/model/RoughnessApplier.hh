//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/RoughnessApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayOperators.hh"
#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
template<class T>
struct RoughnessApplier
{
    T const& executor_builder;
    inline CELER_FUNCTION void operator()(CoreTrackView& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
template<class T>
CELER_FUNCTION void RoughnessApplier<T>::operator()(CoreTrackView& track) const
{
    auto s_physics = track.surface_physics();
    auto model_view = track.surface_model(SurfacePhysicsOrder::roughness);
    auto rng = track.rng();

    auto normal = s_physics.global_normal();
    if (model_view.direction() == SubsurfaceDirection::reverse)
    {
        normal = -normal;
    }

    auto sample = executor_builder(model_view, track.geometry().dir(), normal);

    s_physics.facet_normal(sample(rng));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
