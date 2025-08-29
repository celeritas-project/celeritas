//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/surface/GaussianRoughnessSampler.hh"
#include "celeritas/optical/surface/SurfaceModelView.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "GaussianRoughnessData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct GaussianRoughnessExecutorBuilder
{
    //!@{
    //! \name Type aliases
    using Sampler = EnteringSurfaceNormalSampler<GaussianRoughnessSampler>;
    //!@}

    NativeCRef<GaussianRoughnessData> data;

    CELER_FUNCTION Sampler operator()(SurfaceModelView const& model,
                                      Real3 const& dir,
                                      Real3 const& normal) const
    {
        return Sampler(
            dir, normal, data.sigma_alpha[model.internal_surface_id()]);
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
