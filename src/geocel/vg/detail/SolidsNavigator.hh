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

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "geocel/vg/VecgeomTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Adapt VecGeom's solid model navigation to Celeritas navigation state.
 *
 * Host and device queries use VecGeom's IndexedBVH-backed BVHNavigator
 * directly on the stored VecGeom navigation state, which retains the boundary
 * flag between calls. The last exited volume is retained only from
 * \c ComputeStepAndNextVolume until it is used and cleared by
 * \c RelocateToNextVolume .
 *
 * \deprecated This is the only model supported for vecgeom 2.x and will be
 * replaced by using \c VgView.
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
    // Find the distance out of the state's top volume, ignoring daughters
    CELER_FUNCTION static vg_real_type ComputeExitStep(
        VgReal3 const& glpos, VgReal3 const& gldir, NavState const& state)
    {
        vecgeom::Transformation3D trans;
        state.TopMatrix(trans);
        vg_real_type step = state.Top()->DistanceToOut(
            trans.Transform(glpos), trans.TransformDirection(gldir));
        // Negative if outside, infinite if not found
        return (step > 0 && step < vecgeom::kInfLength) ? step : 0;
    }

    //-----------------------------------------------------------------------//
    // Exit the top volume from its boundary in preparation for relocation
    CELER_FUNCTION static void ExitToParent(VgReal3 const& glpos,
                                            VgReal3 const& gldir,
                                            NavState const& in_state,
                                            NavState& out_state)
    {
        CELER_EXPECT(in_state.Top());
        vecgeom::Transformation3D trans;
        in_state.TopMatrix(trans);
        // Push the point inside the next volume: parents may share the surface
        VgReal3 pos = trans.Transform(
            glpos + vecgeom::BVHNavigator::kBoundaryPush * gldir);

        out_state = in_state;
        VgPlacedVol const* vol = out_state.Top();
        do
        {
            out_state.SetLastExited();
            out_state.Pop();
            pos = vol->GetTransformation()->InverseTransform(pos);
            vol = out_state.Top();
        } while (
            vol
            && (vol->IsAssembly()
                || vol->GetUnplacedVolume()->Inside(pos) != vecgeom::kInside));
        out_state.SetBoundaryState(true);
    }

    //-----------------------------------------------------------------------//
    // Find the deepest volume inside the top volume that contains the point
    CELER_FUNCTION static void LocatePointBelow(VgReal3 const& glpos,
                                                NavState& state)
    {
        CELER_EXPECT(state.Top());
        vecgeom::Transformation3D trans;
        state.TopMatrix(trans);
        vecgeom::BVHNavigator::LocatePointInNavState(
            trans.Transform(glpos), state, /* top = */ false);
        while (state.Top()->IsAssembly())
        {
            state.Pop();
        }
        state.SetBoundaryState(false);
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

        // The exited volume only applies to this crossing: clear it so that a
        // later crossing (e.g., after reflection) may reenter it, and so that
        // entering a daughter never inherits an unrelated excluded volume
        out_state.SetLastExited(decltype(out_state.GetLastExitedState()){});
    }

    //-----------------------------------------------------------------------//
    // Compute the global boundary normal pointing from current to next state.
    CELER_FUNCTION static bool ComputeNextNormal(VgReal3 const& globalpoint,
                                                 NavState const& current,
                                                 NavState const& next,
                                                 VgReal3& normal)
    {
        return vecgeom::BVHNavigator::ComputeNextNormal(
            globalpoint, current, next, normal);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
