//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-FileCopyrightText: 2025 Celeritas contributors
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file geocel/vg/detail/BVHNavigator.hh
 * \brief Bounding Volume Hierarchy navigator directly derived from AdePT
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/bafab78519faafde0b8e5055128c2a3610d43d77/base/inc/AdePT/BVHNavigator.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/NavStateFwd.h>
#include <VecGeom/navigation/NavigationState.h>

#include "corecel/math/Algorithms.hh"

#ifdef VECGEOM_ENABLE_CUDA
#    include <VecGeom/backend/cuda/Interface.h>
#endif

#if CELER_VGNAV == CELER_VGNAV_PATH
#    include <VecGeom/navigation/NavStatePath.h>
#else
#    include "VgNavStateWrapper.hh"
#endif

#include "corecel/Macros.hh"
#include "geocel/vg/VecgeomTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
class BVHNavigator
{
  public:
    using VgPlacedVol = VgPlacedVolume<MemSpace::native>;
#if CELER_VGNAV == CELER_VGNAV_PATH
    using NavState = vecgeom::NavStatePath;
#else
    using NavState = detail::VgNavStateWrapper;
#endif

    static constexpr vg_real_type kBoundaryPush = 10 * vecgeom::kTolerance;

    //! Update path (which must be reset in advance)
    CELER_FUNCTION static void
    LocatePointIn(VgPlacedVol const* vol,
                  VgReal3 const& point,
                  NavState& path,
                  bool top,
                  VgPlacedVol const* exclude = nullptr)
    {
        if (top)
        {
            CELER_ASSERT(vol != nullptr);
            if (!vol->UnplacedContains(point))
            {
                path.Clear();
                return;
            }
        }

        path.Push(vol);

        VgReal3 currentpoint(point);

        while (vol->GetDaughters().size() > 0)
        {
            auto* bvh
                = vecgeom::BVHManager::GetBVH(vol->GetLogicalVolume()->id());
            CELER_ASSERT(bvh);

            // Note: vol *and* daughterlocalpoint are updated by this call
            VgReal3 daughterlocalpoint;
            vol = nullptr;
            if (!bvh->LevelLocate(
                    exclude, currentpoint, vol, daughterlocalpoint))
            {
                // Not inside any daughter
                break;
            }

            currentpoint = daughterlocalpoint;
            path.Push(vol);
            // Only exclude the placed volume once since we could enter it
            // again via a different volume history.
            exclude = nullptr;
        }
    }

    CELER_FUNCTION static void
    RelocatePoint(VgReal3 const& localpoint, NavState& path)
    {
        VgPlacedVol const* currentmother = path.Top();
        VgReal3 transformed = localpoint;
        do
        {
            path.Pop();
            transformed = currentmother->GetTransformation()->InverseTransform(
                transformed);
            currentmother = path.Top();
        } while (currentmother
                 && (currentmother->IsAssembly()
                     || !currentmother->UnplacedContains(transformed)));

        if (currentmother)
        {
            path.Pop();
            return LocatePointIn(currentmother, transformed, path, false);
        }
    }

  private:
    // Computes a step in the current volume from the localpoint into localdir,
    // taking step_limit into account. If a volume is hit, the function calls
    // out_state.SetBoundaryState(true) and hitcandidate is set to the hit
    // daughter volume, or kept unchanged if the current volume is left.
    CELER_FUNCTION static double
    ComputeStepAndHit(VgReal3 const& localpoint,
                      VgReal3 const& localdir,
                      vg_real_type step_limit,
                      NavState const& in_state,
                      NavState& out_state,
                      VgPlacedVol const*& hitcandidate)
    {
        if (step_limit <= 0)
        {
            // We don't need to ask any solid, step not limited by geometry.
            in_state.CopyTo(&out_state);
            out_state.SetBoundaryState(false);
            return 0;
        }

        vg_real_type step = step_limit;
        VgPlacedVol const* pvol = in_state.Top();

        // need to calc DistanceToOut first
        step = pvol->DistanceToOut(localpoint, localdir, step_limit);

        if (step < 0)
            step = 0;

        if (pvol->GetDaughters().size() > 0)
        {
            auto bvh
                = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());
            bvh->CheckDaughterIntersections(
                localpoint, localdir, step, pvol, hitcandidate);
        }

        // now we have the candidates and we prepare the out_state
        in_state.CopyTo(&out_state);
        if (step == vecgeom::kInfLength && step_limit > 0)
        {
            out_state.SetBoundaryState(true);
            do
            {
                out_state.Pop();
            } while (out_state.Top()->IsAssembly());

            return vecgeom::kTolerance;
        }

        // Is geometry further away than physics step?
        if (step > step_limit)
        {
            // Then this is a phyics step and we don't need to do anything.
            out_state.SetBoundaryState(false);
            return step_limit;
        }

        // Otherwise it is a geometry step and we push the point to the
        // boundary.
        out_state.SetBoundaryState(true);

        if (step < 0)
        {
            step = 0;
        }

        return step;
    }

    // Computes a step in the current volume from the localpoint into localdir,
    // until the next daughter bounding box, taking step_limit into account.
    CELER_FUNCTION static double ApproachNextVolume(VgReal3 const& localpoint,
                                                    VgReal3 const& localdir,
                                                    vg_real_type step_limit,
                                                    NavState const& in_state)
    {
        vg_real_type step = step_limit;
        VgPlacedVol const* pvol = in_state.Top();

        if (pvol->GetDaughters().size() > 0)
        {
            auto bvh
                = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());
            // bvh->CheckDaughterIntersections(localpoint, localdir, step,
            // pvol, hitcandidate);
            bvh->ApproachNextDaughter(localpoint, localdir, step, pvol);
            // Make sure we don't "step" on next boundary
            step -= 10 * vecgeom::kTolerance;
        }

        if (step == vecgeom::kInfLength && step_limit > 0)
            return 0;

        // Is geometry further away than physics step?
        if (step > step_limit)
        {
            // Then this is a phyics step and we don't need to do anything.
            return step_limit;
        }

        if (step < 0)
        {
            step = 0;
        }

        return step;
    }

  public:
    // Computes the isotropic safety from the globalpoint.
    CELER_FUNCTION static double
    ComputeSafety(VgReal3 const& globalpoint,
                  NavState const& state,
                  vg_real_type safety
                  = std::numeric_limits<vg_real_type>::infinity())
    {
        VgPlacedVol const* pvol = state.Top();
        vecgeom::Transformation3D m;
        state.TopMatrix(m);
        VgReal3 localpoint = m.Transform(globalpoint);

        // need to calc DistanceToOut first
        safety = min(safety, pvol->SafetyToOut(localpoint));

        if (safety > 0 && pvol->GetDaughters().size() > 0)
        {
            auto bvh
                = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());
            safety = bvh->ComputeSafety(localpoint, safety);
        }

        return safety;
    }

    // Computes a step from the globalpoint (which must be in the current
    // volume) into globaldir, taking step_limit into account. If a volume is
    // hit, the function calls out_state.SetBoundaryState(true) and relocates
    // the state to the next volume.
    CELER_FUNCTION static double
    ComputeStepAndPropagatedState(VgReal3 const& globalpoint,
                                  VgReal3 const& globaldir,
                                  vg_real_type step_limit,
                                  NavState const& in_state,
                                  NavState& out_state,
                                  vg_real_type push = 0)
    {
        // If we are on the boundary, push a bit more
        if (in_state.IsOnBoundary())
        {
            push += kBoundaryPush;
        }
        if (step_limit < push)
        {
            // Go as far as the step limit says, assuming there is no boundary.
            // TODO: Does this make sense?
            in_state.CopyTo(&out_state);
            out_state.SetBoundaryState(false);
            return step_limit;
        }
        step_limit -= push;

        // calculate local point/dir from global point/dir
        VgReal3 localpoint;
        VgReal3 localdir;
        // Impl::DoGlobalToLocalTransformation(in_state, globalpoint,
        // globaldir, localpoint, localdir);
        vecgeom::Transformation3D m;
        in_state.TopMatrix(m);
        localpoint = m.Transform(globalpoint);
        localdir = m.TransformDirection(globaldir);
        // The user may want to move point from boundary before computing the
        // step
        localpoint += push * localdir;

        VgPlacedVol const* hitcandidate = nullptr;
        vg_real_type step = ComputeStepAndHit(
            localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
        step += push;

        if (out_state.IsOnBoundary())
        {
            // Relocate the point after the step to refine out_state.
            localpoint += (step + kBoundaryPush) * localdir;

            if (!hitcandidate)
            {
                // We didn't hit a daughter but instead we're exiting the
                // current volume.
                RelocatePoint(localpoint, out_state);
            }
            else
            {
                // Otherwise check if we're directly entering other daughters
                // transitively.
                localpoint
                    = hitcandidate->GetTransformation()->Transform(localpoint);
                LocatePointIn(hitcandidate, localpoint, out_state, false);
            }

            if (out_state.Top() != nullptr)
            {
                while (out_state.Top()->IsAssembly()
                       || out_state.HasSamePathAsOther(in_state))
                {
                    out_state.Pop();
                }
                CELER_ASSERT(!out_state.Top()
                                  ->GetLogicalVolume()
                                  ->GetUnplacedVolume()
                                  ->IsAssembly());
            }
        }

        return step;
    }

    // Computes a step from the globalpoint (which must be in the current
    // volume) into globaldir, taking step_limit into account. If a volume is
    // hit, the function calls out_state.SetBoundaryState(true) and
    //  - removes all volumes from out_state if the current volume is left, or
    //  - adds the hit daughter volume to out_state if one is hit.
    // However the function does _NOT_ relocate the state to the next volume,
    // that is entering multiple volumes that share a boundary.
    CELER_FUNCTION static double
    ComputeStepAndNextVolume(VgReal3 const& globalpoint,
                             VgReal3 const& globaldir,
                             vg_real_type step_limit,
                             NavState const& in_state,
                             NavState& out_state,
                             vg_real_type push = 0)
    {
        // If we are on the boundary, push a bit more
        if (in_state.IsOnBoundary())
        {
            push += kBoundaryPush;
        }
        if (step_limit < push)
        {
            // Go as far as the step limit says, assuming there is no boundary.
            // TODO: Does this make sense?
            in_state.CopyTo(&out_state);
            out_state.SetBoundaryState(false);
            return step_limit;
        }
        step_limit -= push;

        // calculate local point/dir from global point/dir
        VgReal3 localpoint;
        VgReal3 localdir;
        // Impl::DoGlobalToLocalTransformation(in_state, globalpoint,
        // globaldir, localpoint, localdir);
        vecgeom::Transformation3D m;
        in_state.TopMatrix(m);
        localpoint = m.Transform(globalpoint);
        localdir = m.TransformDirection(globaldir);
        // The user may want to move point from boundary before computing the
        // step
        localpoint += push * localdir;

        VgPlacedVol const* hitcandidate = nullptr;
        vg_real_type step = ComputeStepAndHit(
            localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
        step += push;

        if (out_state.IsOnBoundary())
        {
            if (!hitcandidate)
            {
                VgPlacedVol const* currentmother = out_state.Top();
                VgReal3 transformed = localpoint;
                // Push the point inside the next volume.
                transformed += (step + kBoundaryPush) * localdir;

                do
                {
                    out_state.SetLastExited();
                    out_state.Pop();
                    transformed
                        = currentmother->GetTransformation()->InverseTransform(
                            transformed);
                    currentmother = out_state.Top();
                } while (currentmother
                         && (currentmother->IsAssembly()
                             || !currentmother->UnplacedContains(transformed)));
            }
            else
            {
                out_state.Push(hitcandidate);
            }
        }

        return step;
    }

    // Computes a step from the globalpoint (which must be in the current
    // volume) into globaldir, taking step_limit into account.
    CELER_FUNCTION static vg_real_type
    ComputeStepToApproachNextVolume(VgReal3 const& globalpoint,
                                    VgReal3 const& globaldir,
                                    vg_real_type step_limit,
                                    NavState const& in_state)
    {
        // calculate local point/dir from global point/dir
        VgReal3 localpoint;
        VgReal3 localdir;
        // Impl::DoGlobalToLocalTransformation(in_state, globalpoint,
        // globaldir, localpoint, localdir);
        vecgeom::Transformation3D m;
        in_state.TopMatrix(m);
        localpoint = m.Transform(globalpoint);
        localdir = m.TransformDirection(globaldir);

        vg_real_type step
            = ApproachNextVolume(localpoint, localdir, step_limit, in_state);

        return step;
    }

    // Relocate a state that was returned from ComputeStepAndNextVolume: It
    // recursively locates the pushed point in the containing volume.
    CELER_FUNCTION static void RelocateToNextVolume(VgReal3 const& globalpoint,
                                                    VgReal3 const& globaldir,
                                                    NavState& state)
    {
        // Push the point inside the next volume.
        VgReal3 pushed = globalpoint + kBoundaryPush * globaldir;

        // Calculate local point from global point.
        vecgeom::Transformation3D m;
        state.TopMatrix(m);
        VgReal3 localpoint = m.Transform(pushed);

        VgPlacedVol const* pvol = state.Top();

        state.Pop();
        LocatePointIn(pvol, localpoint, state, false, state.GetLastExited());

        if (state.Top() != nullptr)
        {
            while (state.Top()->IsAssembly())
            {
                state.Pop();
            }
            CELER_ASSERT(!state.Top()
                              ->GetLogicalVolume()
                              ->GetUnplacedVolume()
                              ->IsAssembly());
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
