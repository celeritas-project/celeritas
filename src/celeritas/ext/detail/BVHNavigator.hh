//----------------------------------*-C++-*----------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file BVHNavigator.hh
 * \brief Bounding Volume Hierarchy navigator directly derived from AdePT
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/586a9c43915818bdf764b7315c3dbde7e63a34b8/base/inc/AdePT/BVHNavigator.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/NavStateFwd.h>

#ifdef VECGEOM_ENABLE_CUDA
#    include <VecGeom/backend/cuda/Interface.h>
#endif
#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
class BVHNavigator
{
  public:
    using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const*;

    CELER_FUNCTION static VPlacedVolumePtr_t
    LocatePointIn(vecgeom::VPlacedVolume const*                vol,
                  vecgeom::Vector3D<vecgeom::Precision> const& point,
                  vecgeom::NavigationState&                    path,
                  bool                                         top,
                  vecgeom::VPlacedVolume const* exclude = nullptr)
    {
        if (top)
        {
            assert(vol != nullptr);
            if (!vol->UnplacedContains(point))
                return nullptr;
        }

        path.Push(vol);

        vecgeom::Vector3D<vecgeom::Precision> currentpoint(point);
        vecgeom::Vector3D<vecgeom::Precision> daughterlocalpoint;

        for (auto v = vol; v->GetDaughters().size() > 0;)
        {
            auto bvh = vecgeom::BVHManager::GetBVH(v->GetLogicalVolume()->id());

            if (!bvh->LevelLocate(exclude, currentpoint, v, daughterlocalpoint))
                break;

            currentpoint = daughterlocalpoint;
            path.Push(v);
            // Only exclude the placed volume once since we could enter it
            // again via a different volume history.
            exclude = nullptr;
        }

        return path.Top();
    }

    CELER_FUNCTION static VPlacedVolumePtr_t
    RelocatePoint(vecgeom::Vector3D<vecgeom::Precision> const& localpoint,
                  vecgeom::NavigationState&                    path)
    {
        vecgeom::VPlacedVolume const*         currentmother = path.Top();
        vecgeom::Vector3D<vecgeom::Precision> transformed   = localpoint;
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
        return currentmother;
    }

  private:
    // Computes a step in the current volume from the localpoint into localdir,
    // taking step_limit into account. If a volume is hit, the function calls
    // out_state.SetBoundaryState(true) and hitcandidate is set to the hit
    // daughter volume, or kept unchanged if the current volume is left.
    CELER_FUNCTION static double
    ComputeStepAndHit(vecgeom::Vector3D<vecgeom::Precision> const& localpoint,
                      vecgeom::Vector3D<vecgeom::Precision> const& localdir,
                      vecgeom::Precision                           step_limit,
                      vecgeom::NavigationState const&              in_state,
                      vecgeom::NavigationState&                    out_state,
                      VPlacedVolumePtr_t& hitcandidate)
    {
        vecgeom::Precision step = step_limit;
        VPlacedVolumePtr_t pvol = in_state.Top();

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
        if (step == vecgeom::kInfLength && step_limit > 0.)
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

        if (step < 0.)
        {
            step = 0.;
        }

        return step;
    }

    // Computes a step in the current volume from the localpoint into localdir,
    // until the next daughter bounding box, taking step_limit into account.
    CELER_FUNCTION static double
    ApproachNextVolume(vecgeom::Vector3D<vecgeom::Precision> const& localpoint,
                       vecgeom::Vector3D<vecgeom::Precision> const& localdir,
                       vecgeom::Precision                           step_limit,
                       vecgeom::NavigationState const&              in_state)
    {
        vecgeom::Precision step = step_limit;
        VPlacedVolumePtr_t pvol = in_state.Top();

        if (pvol->GetDaughters().size() > 0)
        {
            auto bvh
                = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());
            // bvh->CheckDaughterIntersections(localpoint, localdir, step,
            // pvol, hitcandidate);
            bvh->ApproachNextDaughter(localpoint, localdir, step, pvol);
            // Make sure we don't "step" on next boundary
            step -= 10. * vecgeom::kTolerance;
        }

        if (step == vecgeom::kInfLength && step_limit > 0.)
            return 0.;

        // Is geometry further away than physics step?
        if (step > step_limit)
        {
            // Then this is a phyics step and we don't need to do anything.
            return step_limit;
        }

        if (step < 0.)
        {
            step = 0.;
        }

        return step;
    }

  public:
    // Computes the isotropic safety from the globalpoint.
    CELER_FUNCTION static double
    ComputeSafety(vecgeom::Vector3D<vecgeom::Precision> const& globalpoint,
                  vecgeom::NavigationState const&              state)
    {
        VPlacedVolumePtr_t        pvol = state.Top();
        vecgeom::Transformation3D m;
        state.TopMatrix(m);
        vecgeom::Vector3D<vecgeom::Precision> localpoint
            = m.Transform(globalpoint);

        // need to calc DistanceToOut first
        vecgeom::Precision safety = pvol->SafetyToOut(localpoint);

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
    CELER_FUNCTION static double ComputeStepAndPropagatedState(
        vecgeom::Vector3D<vecgeom::Precision> const& globalpoint,
        vecgeom::Vector3D<vecgeom::Precision> const& globaldir,
        vecgeom::Precision                           step_limit,
        vecgeom::NavigationState const&              in_state,
        vecgeom::NavigationState&                    out_state,
        vecgeom::Precision                           push = 0.)
    {
        constexpr vecgeom::Precision kPush = 10. * vecgeom::kTolerance;
        // calculate local point/dir from global point/dir
        vecgeom::Vector3D<vecgeom::Precision> localpoint;
        vecgeom::Vector3D<vecgeom::Precision> localdir;
        // Impl::DoGlobalToLocalTransformation(in_state, globalpoint,
        // globaldir, localpoint, localdir);
        vecgeom::Transformation3D m;
        in_state.TopMatrix(m);
        localpoint = m.Transform(globalpoint);
        localdir   = m.TransformDirection(globaldir);
        // The user may want to move point from boundary before computing the
        // step
        localpoint += push * localdir;

        VPlacedVolumePtr_t hitcandidate = nullptr;
        vecgeom::Precision step         = ComputeStepAndHit(
            localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
        if (step < step_limit)
            step += push;

        if (out_state.IsOnBoundary())
        {
            // Relocate the point after the step to refine out_state.
            localpoint += (step + kPush) * localdir;

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
                assert(!out_state.Top()
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
    CELER_FUNCTION static double ComputeStepAndNextVolume(
        vecgeom::Vector3D<vecgeom::Precision> const& globalpoint,
        vecgeom::Vector3D<vecgeom::Precision> const& globaldir,
        vecgeom::Precision                           step_limit,
        vecgeom::NavigationState const&              in_state,
        vecgeom::NavigationState&                    out_state,
        vecgeom::Precision                           push = 0.)
    {
        // calculate local point/dir from global point/dir
        vecgeom::Vector3D<vecgeom::Precision> localpoint;
        vecgeom::Vector3D<vecgeom::Precision> localdir;
        // Impl::DoGlobalToLocalTransformation(in_state, globalpoint,
        // globaldir, localpoint, localdir);
        vecgeom::Transformation3D m;
        in_state.TopMatrix(m);
        localpoint = m.Transform(globalpoint);
        localdir   = m.TransformDirection(globaldir);
        // The user may want to move point from boundary before computing the
        // step
        localpoint += push * localdir;

        VPlacedVolumePtr_t hitcandidate = nullptr;
        vecgeom::Precision step         = ComputeStepAndHit(
            localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
        if (step < step_limit)
            step += push;

        if (out_state.IsOnBoundary())
        {
            if (!hitcandidate)
            {
                vecgeom::VPlacedVolume const* currentmother = out_state.Top();
                vecgeom::Vector3D<vecgeom::Precision> transformed = localpoint;
                // Push the point inside the next volume.
                static constexpr double kPush = 10. * vecgeom::kTolerance;
                transformed += (step + kPush) * localdir;

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
    CELER_FUNCTION static vecgeom::Precision ComputeStepToApproachNextVolume(
        vecgeom::Vector3D<vecgeom::Precision> const& globalpoint,
        vecgeom::Vector3D<vecgeom::Precision> const& globaldir,
        vecgeom::Precision                           step_limit,
        vecgeom::NavigationState const&              in_state)
    {
        // calculate local point/dir from global point/dir
        vecgeom::Vector3D<vecgeom::Precision> localpoint;
        vecgeom::Vector3D<vecgeom::Precision> localdir;
        // Impl::DoGlobalToLocalTransformation(in_state, globalpoint,
        // globaldir, localpoint, localdir);
        vecgeom::Transformation3D m;
        in_state.TopMatrix(m);
        localpoint = m.Transform(globalpoint);
        localdir   = m.TransformDirection(globaldir);

        vecgeom::Precision step
            = ApproachNextVolume(localpoint, localdir, step_limit, in_state);

        return step;
    }

    // Relocate a state that was returned from ComputeStepAndNextVolume: It
    // recursively locates the pushed point in the containing volume.
    CELER_FUNCTION static void
    RelocateToNextVolume(vecgeom::Vector3D<vecgeom::Precision>& globalpoint,
                         vecgeom::Vector3D<vecgeom::Precision> const& globaldir,
                         vecgeom::NavigationState&                    state)
    {
        // Push the point inside the next volume.
        static constexpr double kPush = 10. * vecgeom::kTolerance;
        globalpoint += kPush * globaldir;

        // Calculate local point from global point.
        vecgeom::Transformation3D m;
        state.TopMatrix(m);
        vecgeom::Vector3D<vecgeom::Precision> localpoint
            = m.Transform(globalpoint);

        VPlacedVolumePtr_t pvol = state.Top();

        state.Pop();
        LocatePointIn(pvol, localpoint, state, false, state.GetLastExited());

        if (state.Top() != nullptr)
        {
            while (state.Top()->IsAssembly())
            {
                state.Pop();
            }
            assert(!state.Top()
                        ->GetLogicalVolume()
                        ->GetUnplacedVolume()
                        ->IsAssembly());
        }
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
