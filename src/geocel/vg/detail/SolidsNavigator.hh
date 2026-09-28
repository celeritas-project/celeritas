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

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Adapt VecGeom's solid navigation to Celeritas navigation state.
 *
 * Host and device queries use VecGeom's IndexedBVH-backed BVHNavigator
 * directly on the stored VecGeom navigation state, which retains the boundary
 * flag and the last exited volume between calls.
 */
class SolidsNavigator
{
  public:
    using VgPlacedVol = VgPlacedVolume<MemSpace::native>;
    using NavState = VgNavState;

    //-----------------------------------------------------------------------//
    // Locate a point in the geometry hierarchy
    CELER_FUNCTION static void LocatePointIn(
        VgPlacedVol const* vol,
        VgReal3 const& point,
        NavState& nav,
        bool top,
        VgPlacedVol const* exclude = nullptr)
    {
        vecgeom::BVHNavigator::LocatePointIn(vol, point, nav, top, exclude);

        // Location alone does not establish a crossed boundary. In particular,
        // do not push a newly initialized track past a nearby surface.
        nav.SetBoundaryState(false);
    }

    //-----------------------------------------------------------------------//
    // Find the next boundary and prepare the output state for relocation
    CELER_FUNCTION static vg_real_type ComputeStepAndNextVolume(
        VgReal3 const& glpos,
        VgReal3 const& gldir,
        vg_real_type step_limit,
        NavState const& in_state,
        NavState& out_state)
    {
        // VecGeom treats sub-tolerance step limits as boundary crossings.
        // Query beyond its boundary push, then apply the physics limit here.
        auto query_limit = vecCore::math::Max(
            step_limit, 2 * vecgeom::BVHNavigator::kBoundaryPush);
        auto step = vecgeom::BVHNavigator::ComputeStepAndNextVolume(
            glpos, gldir, query_limit, in_state, out_state);
        if (step > step_limit)
        {
            out_state = in_state;
            out_state.SetBoundaryState(false);
            return step_limit;
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
    CELER_FUNCTION static void RelocateToNextVolume(VgReal3 const& glpos,
                                                    VgReal3 const& gldir,
                                                    NavState const&,
                                                    NavState& out_state)
    {
        // The last exited volume was recorded in the output state by
        // ComputeStepAndNextVolume, preventing reentry at the exact boundary
        vecgeom::BVHNavigator::RelocateToNextVolume(glpos, gldir, out_state);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
