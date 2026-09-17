//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/SolidsNavigator.hh
//! \sa geocel/vg/Vecgeom.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <VecGeom/navigation/BVHNavigator.h>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "geocel/vg/VecgeomTypes.hh"

#include "ScopedVgNavState.hh"
#include "VgNavStateWrapper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Adapt VecGeom's solid navigation to Celeritas navigation state.
 *
 * Host and device queries use VecGeom's IndexedBVH-backed BVHNavigator.
 * The scoped state adapters copy the resulting navigation path back to
 * Celeritas after each query.
 */
class SolidsNavigator
{
  public:
    using VgPlacedVol = VgPlacedVolume<MemSpace::native>;

#if CELER_VGNAV == CELER_VGNAV_PATH
    using NavState = vecgeom::NavStatePath;
#else
    using NavState = detail::VgNavStateWrapper;
#endif

    //-----------------------------------------------------------------------//
    // Locate a point in the geometry hierarchy
    CELER_FUNCTION static void LocatePointIn(
        VgPlacedVol const* vol,
        VgReal3 const& point,
        NavState& nav,
        bool top,
        VgPlacedVol const* exclude = nullptr)
    {
        ScopedVgNavState temp_nav{nav};
        vecgeom::BVHNavigator::LocatePointIn(
            vol, point, temp_nav, top, exclude);

        // Location alone does not establish a crossed boundary. In particular,
        // do not push a newly initialized track past a nearby surface.
        VgNavState& located = temp_nav;
        located.SetBoundaryState(false);
    }

    //-----------------------------------------------------------------------//
    // Find the next boundary and prepare the fully relocated output state
    CELER_FUNCTION static vg_real_type ComputeStepAndNextVolume(
        VgReal3 const& glpos,
        VgReal3 const& gldir,
        vg_real_type step_limit,
        NavState const& in_state,
        NavState& out_state)
    {
        ScopedVgNavState temp_out_state{out_state};
        // VecGeom treats sub-tolerance step limits as boundary crossings.
        // Query beyond its boundary push, then apply the physics limit here.
        auto query_limit = vecCore::math::Max(
            step_limit, 2 * vecgeom::BVHNavigator::kBoundaryPush);
        auto step = vecgeom::BVHNavigator::ComputeStepAndNextVolume(
            glpos, gldir, query_limit, in_state, temp_out_state);
        if (step > step_limit)
        {
            VgNavState& next = temp_out_state;
            next = in_state;
            next.SetBoundaryState(false);
            return step_limit;
        }

        // Keep VecGeom's last-exited volume until relocation is complete:
        // Celeritas's compact state stores only the path and boundary flag.
        VgNavState& next = temp_out_state;
        if (next.IsOnBoundary() && !next.IsOutside())
        {
            vecgeom::BVHNavigator::RelocateToNextVolume(
                glpos + step * gldir, gldir, next);
        }
        return step;
    }

    //-----------------------------------------------------------------------//
    // Computes the isotropic safety from the globalpoint
    CELER_FUNCTION static double ComputeSafety(
        VgReal3 const& glpos,
        NavState const& curr,
        vg_real_type safety = std::numeric_limits<vg_real_type>::infinity())
    {
        real_type result
            = vecgeom::BVHNavigator::ComputeSafety(glpos, curr, safety);
        result = vecCore::math::Min(result, safety);

        return result;
    }

    //-----------------------------------------------------------------------//
    // Relocate a state that was returned from ComputeStepAndNextVolume
    CELER_FUNCTION static void RelocateToNextVolume(
        VgReal3 const&, VgReal3 const&, NavState&)
    {
        // The output state was relocated before discarding temporary metadata
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
