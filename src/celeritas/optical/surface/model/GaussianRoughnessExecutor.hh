//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/surface/GaussianRoughnessSampler.hh"

#include "GaussianRoughnessData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample and save a facet normal for the Gaussian roughness model.
 */
struct GaussianRoughnessExecutor
{
    NativeCRef<GaussianRoughnessData> data;

    //! Apply Gaussian roughness executor
    CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        auto s_phys = track.surface_physics();
        auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::roughness)
                                .internal_surface_id();
        CELER_ASSERT(sub_model_id < data.sigma_alpha.size());
        EnteringSurfaceNormalSampler<GaussianRoughnessSampler> sample_facet{
            track.geometry().dir(),
            s_phys.global_normal(),
            data.sigma_alpha[sub_model_id]};
        auto rng = track.rng();
        s_phys.facet_normal(sample_facet(rng));
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
