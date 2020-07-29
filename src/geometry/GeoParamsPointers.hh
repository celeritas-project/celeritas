//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParamsPointers.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGView_hh
#define geometry_VGView_hh

#include <VecGeom/volumes/PlacedVolume.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Pointers to persistent data used by VecGeom implementation.
 *
 * If the GeoParamsPointers is constructed by \c VGHost::host_view, it points
 * to a \c vecgeom::cxx::VPlacedVolume . If built by \c
 * VGDevice::device_pointers, it points to a \c vecgeom::cuda::VPlacedVolume .
 *
 * Note that because of VecGeom default namespaces triggered by the presence of
 * the \c __NVCC__ macro, this data structure actually has different types
 * <em>depending on what compiler is active</em>. Since the \c GeoTrackView
 * implementation is designed to work with both CPU and GPU (depending on
 * \c __CUDA_ARCH__ and whether the code is on device, rather than the \c
 * __NVCC__ compiler) we can't simply declare this pointer to be `::cuda` or
 * `::cxx`.
 */
struct GeoParamsPointers
{
    const vecgeom::VPlacedVolume* world_volume = nullptr;

    CELER_FUNCTION operator bool() const { return bool(world_volume); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGView_hh
