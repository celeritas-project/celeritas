//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/surface/SmearRoughnessSampler.hh"
#include "celeritas/optical/surface/SurfaceModelView.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "SmearRoughnessData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct a sampling executor for a smear roughness model.
 */
struct SmearRoughnessExecutor
{
    //!@{
    //! \name Type aliases
    using Sampler = EnteringSurfaceNormalSampler<SmearRoughnessSampler>;
    //!@}

    NativeCRef<SmearRoughnessData> data;

    inline CELER_FUNCTION Sampler operator()(SurfaceModelView const& model,
                                             Real3 const& dir,
                                             Real3 const& normal) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct sampler for the given model.
 */
CELER_FUNCTION auto
SmearRoughnessExecutor::operator()(SurfaceModelView const& model,
                                   Real3 const& dir,
                                   Real3 const& normal) const -> Sampler
{
    return Sampler(dir, normal, data.roughness[model.internal_surface_id()]);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
