//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/BVH.h>
#include <VecGeom/base/Config.h>

#include "corecel/Assert.hh"
#if defined(VECGEOM_USE_SURF) && !defined(__NVCC__)
#    include <VecGeom/surfaces/BrepHelper.h>
#endif

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
template<class T>
struct CudaPointers
{
    T* kernel{nullptr};
    T* symbol{nullptr};
};

//---------------------------------------------------------------------------//
// Get pointers to the device BVH after setup, for consistency checking
CudaPointers<vecgeom::cuda::BVH const> bvh_pointers_device();

//---------------------------------------------------------------------------//
#if defined(VECGEOM_USE_SURF) && !defined(__NVCC__)
// Set up surface tracking
void setup_surface_tracking_device(vgbrep::SurfData<vecgeom::Precision> const&);

// Tear down surface tracking
void teardown_surface_tracking_device();
#endif

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#ifndef VECGEOM_ENABLE_CUDA
inline CudaPointers<vecgeom::cuda::BVH const> bvh_pointers_device()
{
    CELER_ASSERT_UNREACHABLE();
}

#    if defined(VECGEOM_USE_SURF) && !defined(__NVCC__)
inline void
setup_surface_tracking_device(vgbrep::SurfData<vecgeom::Precision> const&)
{
    CELER_ASSERT_UNREACHABLE();
}

inline void teardown_surface_tracking_device()
{
    CELER_ASSERT_UNREACHABLE();
}
#    endif
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
