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
#include "corecel/Types.hh"
#include "geocel/vg/VecgeomTypes.hh"

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
    using VgPlacedVol = VgPlacedVolume<MemSpace::native>;
    using NavState = VgNavState;

    //-----------------------------------------------------------------------//
    // Locate a point in the geometry hierarchy
    CELER_FUNCTION static void
    LocatePointIn(VgPlacedVol const* vol,
                  VgReal3 const& point,
                  NavState& nav,
                  bool top,
                  VgPlacedVol const* exclude = nullptr)
    {
        if (exclude)
        {
            // Exclude the volume from the search
            vecgeom::GlobalLocator::LocateGlobalPointExclVolume(
                vol, exclude, point, nav, top);
        }
        else
        {
            // TODO: eliminate this branch by always using Excl
            // Locate the point in the volume hierarchy
            vecgeom::GlobalLocator::LocateGlobalPoint(vol, point, nav, top);
        }
    }

    //-----------------------------------------------------------------------//
    CELER_FUNCTION static vg_real_type
    ComputeStepAndNextVolume(VgReal3 const& glpos,
                             VgReal3 const& gldir,
                             vg_real_type step_limit,
                             NavState const& in_state,
                             NavState& out_state)
    {
        auto* curr_volume = in_state.Top()->GetLogicalVolume();

        // simple dispatch implementation
        auto* navigator = curr_volume->GetNavigator();
        real_type step = navigator->ComputeStepAndPropagatedState(
            glpos, gldir, step_limit, in_state, out_state);
        out_state.SetLastExited({});

        return step;
    }

    //-----------------------------------------------------------------------//
    // Computes the isotropic safety from the globalpoint
    CELER_FUNCTION static double
    ComputeSafety(VgReal3 const& glpos,
                  NavState const& curr,
                  vg_real_type safety
                  = std::numeric_limits<vg_real_type>::infinity())
    {
        auto* navigator = curr.Top()->GetLogicalVolume()->GetNavigator();
        real_type result
            = navigator->GetSafetyEstimator()->ComputeSafety(glpos, curr);
        result = vecCore::math::Min(result, safety);

        return result;
    }

    //-----------------------------------------------------------------------//
    // Relocate a state that was returned from ComputeStepAndNextVolume
    CELER_FUNCTION static void RelocateToNextVolume(VgReal3 const& glpos,
                                                    VgReal3 const& gldir,
                                                    NavState& curr,
                                                    NavState& next)
    {
        VgPlacedVol const* pvol = curr.Top();
        curr.Pop();
        LocatePointIn(curr.Top(), glpos, curr, false, pvol);

        // try to use a point in next volume
        // Push the point inside the next volume.
        static constexpr vg_real_type kBoundaryPush = 10 * vecgeom::kTolerance;
        VgReal3 pushed = glpos + kBoundaryPush * gldir;
        LocatePointIn(next.Top(), pushed, next, false, nullptr);

        if (curr.Top() != nullptr)
        {
            while (curr.Top()->IsAssembly())
            {
                curr.Pop();
            }
            CELER_ASSERT(!curr.Top()
                              ->GetLogicalVolume()
                              ->GetUnplacedVolume()
                              ->IsAssembly());
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
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
