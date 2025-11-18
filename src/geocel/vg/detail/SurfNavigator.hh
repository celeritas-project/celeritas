//---------------------------------------------------------------------------//
// SPDX-FileCopyrightText: 2023 CERN
// SPDX-FileCopyrightText: 2024 Celeritas
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file geocel/vg/detail/SurfNavigator.hh
 * \brief Navigation methods using the surface model.
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/e03b856523164fb13f9f030d52297db96c8a2c8d/base/inc/AdePT/SurfNavigator.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Config.h>

#ifndef VECGEOM_USE_SURF
#    error "VecGeom surface capability required to include this file"
#endif

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>

#ifdef VECGEOM_ENABLE_CUDA
#    include <VecGeom/backend/cuda/Interface.h>
#endif

#include "corecel/Macros.hh"
#include "geocel/vg/VecgeomTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
class SurfNavigator
{
  public:
    using SurfData = vgbrep::SurfData<vg_real_type>;
    using NavState = VgNavState;

    static constexpr vg_real_type kBoundaryPush = 10 * vecgeom::kTolerance;

    /// @brief Locates the point in the geometry volume tree
    /// @param pvol_id Placed volume id to be checked first
    /// @param point Point to be checked, in the local frame of pvol
    /// @param path Path to a parent of pvol that must contain the point
    /// @param top Check first if pvol contains the point
    /// @param exclude Placed volume id to exclude from the search
    /// @return Index of the placed volume that contains the point
    CELER_FUNCTION static VgPlacedVolumeInt
    LocatePointIn(VgPlacedVolumeInt pvol_id,
                  VgReal3 const& point,
                  NavState& nav,
                  bool top,
                  VgPlacedVolumeInt* exclude = nullptr)
    {
        return vgbrep::protonav::BVHSurfNavigator<vg_real_type>::LocatePointIn(
            pvol_id, point, nav, top, exclude);
    }

    /// @brief Computes the isotropic safety from the globalpoint.
    /// @param globalpoint Point in global coordinates
    /// @param state Path where to compute safety
    /// @return Isotropic safe distance
    CELER_FUNCTION static vg_real_type
    ComputeSafety(VgReal3 const& globalpoint, NavState const& state)
    {
        auto safety
            = vgbrep::protonav::BVHSurfNavigator<vg_real_type>::ComputeSafety(
                globalpoint, state);
        return safety;
    }

    // Computes a step from the globalpoint (which must be in the current
    // volume) into globaldir, taking step_limit into account. If a volume is
    // hit, the function calls out_state.SetBoundaryState(true) and relocates
    // the state to the next volume.
    //
    // The surface model does automatic relocation, so this function does it as
    // well.
    CELER_FUNCTION static vg_real_type
    ComputeStepAndNextVolume(VgReal3 const& globalpoint,
                             VgReal3 const& globaldir,
                             vg_real_type step_limit,
                             NavState const& in_state,
                             NavState& out_state,
                             VgSurfaceInt& hitsurf)
    {
        if (step_limit <= 0)
        {
            in_state.CopyTo(&out_state);
            out_state.SetBoundaryState(false);
            return step_limit;
        }

        auto step = vgbrep::protonav::BVHSurfNavigator<
            vg_real_type>::ComputeStepAndNextSurface(globalpoint,
                                                     globaldir,
                                                     in_state,
                                                     out_state,
                                                     hitsurf,
                                                     step_limit);
        return step;
    }

    // Computes a step from the globalpoint (which must be in the current
    // volume) into globaldir, taking step_limit into account. If a volume is
    // hit, the function calls out_state.SetBoundaryState(true) and relocates
    // the state to the next volume.
    CELER_FUNCTION static vg_real_type
    ComputeStepAndPropagatedState(VgReal3 const& globalpoint,
                                  VgReal3 const& globaldir,
                                  vg_real_type step_limit,
                                  VgSurfaceInt& hit_surf,
                                  NavState const& in_state,
                                  NavState& out_state)
    {
        return ComputeStepAndNextVolume(
            globalpoint, globaldir, step_limit, in_state, out_state, hit_surf);
    }

    // Relocate a state that was returned from ComputeStepAndNextVolume: the
    // surface model does this computation within ComputeStepAndNextVolume, so
    // the relocation does nothing
    CELER_FUNCTION static void RelocateToNextVolume(VgReal3 const& globalpoint,
                                                    VgReal3 const& globaldir,
                                                    VgSurfaceInt hitsurf_index,
                                                    NavState& out_state)
    {
        CELER_EXPECT(!out_state.IsOutside());
        vgbrep::CrossedSurface crossed_surf;
        vgbrep::protonav::BVHSurfNavigator<vg_real_type>::RelocateToNextVolume(
            globalpoint,
            globaldir,
            vg_real_type(0),
            hitsurf_index,
            out_state,
            crossed_surf);
    }
};
//---------------------------------------------------------------------------//
}  // namespace detail
}  // End namespace celeritas
