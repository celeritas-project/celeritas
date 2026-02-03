//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/surface/SmearRoughnessSampler.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "SmearRoughnessData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample and save a facet normal for the smear roughness model.
 *
 * TODO: refactor into roughness applier and SmearNormalCalculator
 */
struct SmearRoughnessExecutor
{
    NativeCRef<SmearRoughnessData> data;

    //! Apply smear roughness executor
    CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        auto s_phys = track.surface_physics();
        auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::roughness)
                                .internal_surface_id();
        CELER_ASSERT(sub_model_id < data.roughness.size());
        EnteringSurfaceNormalSampler<SmearRoughnessSampler> sample_facet{
            track.geometry().dir(),
            s_phys.global_normal(),
            data.roughness[sub_model_id]};
        auto rng = track.rng();
        s_phys.facet_normal(sample_facet(rng));
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
