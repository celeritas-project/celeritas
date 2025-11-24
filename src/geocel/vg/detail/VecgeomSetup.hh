//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "geocel/vg/VecgeomTypes.hh"

#if CELERITAS_VECGEOM_SURFACE && !defined(__NVCC__)
#    include <VecGeom/surfaces/BrepHelper.h>
#endif
#include <VecGeom/base/BVH.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
#if VECGEOM_VERSION >= 0x020000
using CudaBVH_t = vecgeom::cuda::BVH<vgbvh_real_type>;
#else
using CudaBVH_t = vecgeom::cuda::BVH;
#endif

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
CudaPointers<detail::CudaBVH_t const> bvh_pointers_device();

//---------------------------------------------------------------------------//
// Get pointers to the global nav index after setup, for consistency checking
CudaPointers<unsigned int const> navindex_pointers_device();

//---------------------------------------------------------------------------//
#if CELERITAS_VECGEOM_SURFACE && !defined(__NVCC__)
// Set up surface tracking
void setup_surface_tracking_device(vgbrep::SurfData<vecgeom::Precision> const&);

// Tear down surface tracking
void teardown_surface_tracking_device();
#endif

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#ifndef VECGEOM_ENABLE_CUDA
inline CudaPointers<CudaBVH_t const> bvh_pointers_device()
{
    CELER_ASSERT_UNREACHABLE();
}

inline CudaPointers<unsigned int const> navindex_pointers_device()
{
    CELER_ASSERT_UNREACHABLE();
}

#    if CELERITAS_VECGEOM_SURFACE && !defined(__NVCC__)
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
