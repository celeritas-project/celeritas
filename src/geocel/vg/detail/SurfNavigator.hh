// SPDX-FileCopyrightText: 2023 CERN
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
#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>

#ifdef VECGEOM_ENABLE_CUDA
#    include <VecGeom/backend/cuda/Interface.h>
#endif
#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
class SurfNavigator
{
  public:
    using Precision = vecgeom::Precision;
    using Vector3D = vecgeom::Vector3D<vecgeom::Precision>;
    using SurfData = vgbrep::SurfData<Precision>;
    using Real_b = typename SurfData::Real_b;
    using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const*;

    static constexpr Precision kBoundaryPush = 10 * vecgeom::kTolerance;

    /// @brief Locates the point in the geometry volume tree
    /// @param pvol_id Placed volume id to be checked first
    /// @param point Point to be checked, in the local frame of pvol
    /// @param path Path to a parent of pvol that must contain the point
    /// @param top Check first if pvol contains the point
    /// @param exclude Placed volume id to exclude from the search
    /// @return Index of the placed volume that contains the point
    CELER_FUNCTION static int LocatePointIn(int pvol_id,
                                            Vector3D const& point,
                                            vecgeom::NavigationState& path,
                                            bool top,
                                            int* exclude = nullptr)
    {
        return vgbrep::protonav::BVHSurfNavigator<Precision>::LocatePointIn(
            pvol_id, point, path, top, exclude);
    }

    /// @brief Computes the isotropic safety from the globalpoint.
    /// @param globalpoint Point in global coordinates
    /// @param state Path where to compute safety
    /// @return Isotropic safe distance
    CELER_FUNCTION static Precision
    ComputeSafety(Vector3D const& globalpoint,
                  vecgeom::NavigationState const& state)
    {
        auto safety
            = vgbrep::protonav::BVHSurfNavigator<Precision>::ComputeSafety(
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
    CELER_FUNCTION static Precision
    ComputeStepAndNextVolume(Vector3D const& globalpoint,
                             Vector3D const& globaldir,
                             Precision step_limit,
                             vecgeom::NavigationState const& in_state,
                             vecgeom::NavigationState& out_state,
                             long& hitsurf)
    {
        if (step_limit <= 0)
        {
            in_state.CopyTo(&out_state);
            out_state.SetBoundaryState(false);
            return step_limit;
        }

        auto step = vgbrep::protonav::BVHSurfNavigator<
            Precision>::ComputeStepAndNextSurface(globalpoint,
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
    CELER_FUNCTION static Precision
    ComputeStepAndPropagatedState(Vector3D const& globalpoint,
                                  Vector3D const& globaldir,
                                  Precision step_limit,
                                  long& hit_surf,
                                  vecgeom::NavigationState const& in,
                                  vecgeom::NavigationState& out)
    {
        return ComputeStepAndNextVolume(
            globalpoint, globaldir, step_limit, in, out, hit_surf);
    }

    // Relocate a state that was returned from ComputeStepAndNextVolume: the
    // surface model does this computation within ComputeStepAndNextVolume, so
    // the relocation does nothing
    CELER_FUNCTION static void
    RelocateToNextVolume(Vector3D const& globalpoint,
                         Vector3D const& globaldir,
                         long hitsurf_index,
                         vecgeom::NavigationState& out_state)
    {
        CELER_EXPECT(!out_state.IsOutside());
        vgbrep::CrossedSurface crossed_surf;
        vgbrep::protonav::BVHSurfNavigator<Precision>::RelocateToNextVolume(
            globalpoint,
            globaldir,
            Precision(0),
            hitsurf_index,
            out_state,
            crossed_surf);
    }
};
//---------------------------------------------------------------------------//
}  // namespace detail
}  // End namespace celeritas
