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
 * Wrapper for surface roughness models.
 *
 * This wrapper determines the relevant roughness model and surface data for
 * the track, and ensures that the normal being sampled from satisfies the
 * entering surface convention. The roughness executor should return a functor
 * that samples the facet normal given an RNG engine.
 */
template<class T>
struct RoughnessApplier
{
    T const& make_sampler;

    inline CELER_FUNCTION void operator()(CoreTrackView& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample and assign facet normal for the track.
 *
 * Ensures that the track's facet normal state field is updated with a facet
 * normal following the entering surface convention.
 */
template<class T>
CELER_FUNCTION void RoughnessApplier<T>::operator()(CoreTrackView& track) const
{
    auto const& track_dir = track.geometry().dir();
    auto s_physics = track.surface_physics();
    auto model_view
        = s_physics.surface_model(track_dir, SurfacePhysicsOrder::roughness);
    auto rng = track.rng();

    // Ensure the local normal follows the entering surface convention
    auto normal = s_physics.global_normal();
    if (model_view.direction() == SubsurfaceDirection::reverse)
    {
        normal = -normal;
    }

    CELER_ASSERT(is_entering_surface(track_dir, normal));

    // Construct normal sampler from executor
    auto sample = make_sampler(model_view, track_dir, normal);

    s_physics.facet_normal(sample(rng));

    CELER_ENSURE(is_entering_surface(track_dir, s_physics.facet_normal()));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
