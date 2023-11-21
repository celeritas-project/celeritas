// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file SurfNavigator.hh
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
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/surfaces/Navigator.h>

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
    using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const*;
    using SurfData = vgbrep::SurfData<Precision>;

    static constexpr Precision kBoundaryPush = 10 * vecgeom::kTolerance;

    // Locates the point in the geometry volume tree
    CELER_FUNCTION static VPlacedVolumePtr_t
    LocatePointIn(VPlacedVolumePtr_t vol,
                  Vector3D const& point,
                  vecgeom::NavStateIndex& path,
                  bool top)
    {
        return vgbrep::protonav::LocatePointIn(vol, point, path, top);
    }

    // Computes the isotropic safety from the globalpoint.
    CELER_FUNCTION static Precision
    ComputeSafety(Vector3D const& globalpoint,
                  vecgeom::NavStateIndex const& state)
    {
        int closest_surf = 0;
        return vgbrep::protonav::ComputeSafety(
            globalpoint, state, closest_surf);
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
                             vecgeom::NavStateIndex const& in_state,
                             vecgeom::NavStateIndex& out_state,
                             [[maybe_unused]] Precision push = 0)
    {
        if (step_limit <= 0)
        {
            in_state.CopyTo(&out_state);
            out_state.SetBoundaryState(false);
            return step_limit;
        }

        int exit_surf = 0;
        auto step = vgbrep::protonav::ComputeStepAndHit(
            globalpoint, globaldir, in_state, out_state, exit_surf, step_limit);
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
                                  vecgeom::NavStateIndex const& in_state,
                                  vecgeom::NavStateIndex& out_state,
                                  Precision push = 0)
    {
        return ComputeStepAndNextVolume(
            globalpoint, globaldir, step_limit, in_state, out_state, push);
    }

    // Relocate a state that was returned from ComputeStepAndNextVolume: the
    // surface model does this computation within ComputeStepAndNextVolume, so
    // the relocation does nothing
    CELER_FUNCTION static void
    RelocateToNextVolume(Vector3D const& /*globalpoint*/,
                         Vector3D const& /*globaldir*/,
                         vecgeom::NavStateIndex& /*state*/)
    {
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // End namespace celeritas
