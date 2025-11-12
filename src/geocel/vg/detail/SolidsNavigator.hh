//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/SolidsNavigator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/BVH.h>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/base/Global.h>
#include <VecGeom/base/Version.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Pointers to device data, obtained from a kernel launch or from runtime.
 *
 * The \c kernel data is copied from inside a kernel to global heap memory, and
 * thence to this result. The \c symbol data is copied via \c
 * cudaMemcpyFromSymbol .
 */
class SolidsNavigator
{
  public:
    using Precision = vecgeom::Precision;
    using Vector3D = vecgeom::Vector3D<vecgeom::Precision>;
    using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const*;

    //-----------------------------------------------------------------------//
    // Locate a point in the geometry hierarchy
    CELER_FUNCTION static VPlacedVolumePtr_t
    LocatePointIn(vecgeom::VPlacedVolume const* vol,
                  Vector3D const& point,
                  vecgeom::NavigationState& path,
                  bool top,
                  vecgeom::VPlacedVolume const* exclude = nullptr)
    {
        if (exclude)
        {
            // Exclude the volume from the search
            vecgeom::GlobalLocator::LocateGlobalPointExclVolume(
                vol, exclude, point, path, top);
        }
        else
        {
            // Locate the point in the volume hierarchy
            vecgeom::GlobalLocator::LocateGlobalPoint(vol, point, path, top);
        }
        return path.Top();
    }

    //-----------------------------------------------------------------------//
    CELER_FUNCTION static Precision
    ComputeStepAndNextVolume(Vector3D const& glpos,
                             Vector3D const& gldir,
                             Precision step_limit,
                             vecgeom::NavigationState& curr,
                             vecgeom::NavigationState& next)
    {
        auto* curr_volume = curr.Top()->GetLogicalVolume();

        //.. This faster method provides the next step, but not the next volume
        // real_type step = navigator->ComputeStep(glpos, gldir, step_limit,
        // curr, next);

        // simple dispatch implementation
        auto* navigator = curr_volume->GetNavigator();
        real_type step = navigator->ComputeStepAndPropagatedState(
            glpos, gldir, step_limit, curr, next);

        return step;
    }

    //-----------------------------------------------------------------------//
    // Computes the isotropic safety from the globalpoint
    CELER_FUNCTION static double
    ComputeSafety(Vector3D const& glpos,
                  vecgeom::NavigationState const& curr,
                  Precision safety = std::numeric_limits<Precision>::infinity())
    {
        // printf("SolidsNav CompSafety() @sp1: glpos=(%f,%f,%f)\n",
        //        glpos.x(), glpos.y(), glpos.z());
        auto* navigator = curr.Top()->GetLogicalVolume()->GetNavigator();
        real_type result
            = navigator->GetSafetyEstimator()->ComputeSafety(glpos, curr);
        result = vecCore::math::Min(result, safety);
        // printf("SolidsNav CompSafety() @sp2: return safety=%f\n", result);
        return result;
    }

    //-----------------------------------------------------------------------//
    // Relocate a state that was returned from ComputeStepAndNextVolume
    CELER_FUNCTION static void
    RelocateToNextVolume(Vector3D const& glpos,
                         Vector3D const& gldir,
                         vecgeom::NavigationState& curr,
                         vecgeom::NavigationState& next)
    {
        VPlacedVolumePtr_t pvol = curr.Top();
        curr.Pop();
        LocatePointIn(curr.Top(), glpos, curr, false, pvol);

        // try to use a point in next volume
        // Push the point inside the next volume.
        static constexpr Precision kBoundaryPush = 10 * vecgeom::kTolerance;
        Vector3D pushed = glpos + kBoundaryPush * gldir;
        LocatePointIn(next.Top(), pushed, next, false, nullptr);

        if (curr.Top() != nullptr)
        {
            while (curr.Top()->IsAssembly())
            {
                curr.Pop();
            }
            assert(!curr.Top()
                        ->GetLogicalVolume()
                        ->GetUnplacedVolume()
                        ->IsAssembly());
        }
    }

    //---------------------------------------------------------------------------//
    CELER_FUNCTION static VPlacedVolumePtr_t
    RelocatePoint(Vector3D const& localpoint, vecgeom::NavigationState& path)
    {
        vecgeom::VPlacedVolume const* currentmother = path.Top();
        Vector3D transformed = localpoint;
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
};
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
