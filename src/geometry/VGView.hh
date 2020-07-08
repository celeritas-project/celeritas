//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGView.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGView_hh
#define geometry_VGView_hh

#include <VecGeom/volumes/PlacedVolume.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * View to persistent data used by VecGeom implementation.
 *
 * If the VGView is constructed by \c VGHost::host_view, it points to a
 * \c vecgeom::cxx::VPlacedVolume . If built by \c VGDevice::device_view, it
 * points to a \c vecgeom::cuda::VPlacedVolume .
 *
 * Note that because of VecGeom default namespaces triggered by the presence of
 * the \c __NVCC__ macro, this data structure actually has different types
 * <em>depending on what compiler is active</em>. Since the \c VGGeometry
 * implementation is designed to work with both CPU and GPU (depending on
 * \c __CUDA_ARCH__ and whether the code is on device, rather than the \c
 * __NVCC__ compiler) we can't simply declare this pointer to be `::cuda` or
 * `::cxx`.
 */
struct VGView
{
    const vecgeom::VPlacedVolume* world_volume = nullptr;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGView_hh
