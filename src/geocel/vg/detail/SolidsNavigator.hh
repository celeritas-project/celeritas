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

    using NavState = detail::VgNavStateWrapper;
    using NavImpl = vecgeom::BVHNavigator;

    //-----------------------------------------------------------------------//
    // Locate a point in the geometry hierarchy
    CELER_FUNCTION static void
    LocatePointIn(VgPlacedVol const* vol,
                  VgReal3 const& localpos,
                  NavState& state,
                  bool top,
                  VgPlacedVol const* exclude = nullptr)
    {
        ScopedVgNavState temp_state{state};
        NavImpl::LocatePointIn(vol, localpos, temp_state, top, exclude);
    }

    //-----------------------------------------------------------------------//
    CELER_FUNCTION static vg_real_type
    ComputeStepAndNextVolume(VgReal3 const& pos,
                             VgReal3 const& dir,
                             vg_real_type step_limit,
                             NavState const& in_state,
                             NavState& out_state)
    {
        ScopedVgNavState temp_state{out_state};
        // Use 1000 * kTolerance like ADePT
        constexpr vg_real_type search_bump{1e-5};
        return NavImpl::ComputeStepAndNextVolume(
            pos, dir, step_limit, in_state, temp_state, search_bump);
    }

    //-----------------------------------------------------------------------//
    // Computes the isotropic safety from the globalpoint
    CELER_FUNCTION static vg_real_type
    ComputeSafety(VgReal3 const& pos,
                  NavState const& state,
                  vg_real_type limit
                  = std::numeric_limits<vg_real_type>::infinity())
    {
        return NavImpl::ComputeSafety(pos, state, limit);
    }

    //-----------------------------------------------------------------------//
    // Relocate a state that was returned from ComputeStepAndNextVolume
    CELER_FUNCTION static void RelocateToNextVolume(VgReal3 const& pos,
                                                    VgReal3 const& dir,
                                                    NavState& state)
    {
        ScopedVgNavState temp_state{state};
        return NavImpl::RelocateToNextVolume(pos, dir, temp_state);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
