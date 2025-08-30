//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
class SurfaceModelView;
//---------------------------------------------------------------------------//
/*!
 * Trivially sample a perfectly polished surface.
 *
 * A perfectly polished surface has the same local facet normal as the global
 * normal.
 */
struct PolishedRoughnessSampler
{
    Real3 const& normal;

    template<class Engine>
    CELER_FUNCTION Real3 operator()(Engine&) const
    {
        return normal;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construct a sampling executor for a polished roughness model.
 */
struct PolishedRoughnessExecutor
{
    //!@{
    //! name Type aliases
    using Sampler = PolishedRoughnessSampler;
    //!@}

    inline CELER_FUNCTION Sampler operator()(SurfaceModelView const&,
                                             Real3 const&,
                                             Real3 const& normal) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct polished roughness sampler.
 */
CELER_FUNCTION auto
PolishedRoughnessExecutor::operator()(SurfaceModelView const&,
                                      Real3 const&,
                                      Real3 const& normal) const -> Sampler
{
    return Sampler{normal};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
