//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/PlacedVolume.h>
#include "base/Array.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Pointers to persistent data used by VecGeom implementation.
 *
 * If the GeoParamsPointers is constructed by \c VGHost::host_pointers, it
 * points to a \c vecgeom::cxx::VPlacedVolume . If built by \c
 * VGDevice::device_pointers, it points to a \c vecgeom::cuda::VPlacedVolume .
 *
 * Note that because of VecGeom default namespaces triggered by the presence of
 * the \c __NVCC__ macro, this data structure actually has different types
 * <em>depending on what compiler is active</em>. Since the \c GeoTrackView
 * implementation is designed to work with both CPU and GPU (depending on
 * \c __CUDA_ARCH__ and whether the code is on device, rather than the \c
 * __NVCC__ compiler) we can't simply declare this pointer to be in the \c cuda
 * or \c cxx explicit namespaces.
 */
struct GeoParamsPointers
{
    const vecgeom::VPlacedVolume* world_volume = nullptr;

    //! Check whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return bool(world_volume);
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Data required to initialize a geometry state.
 */
struct GeoStateInitializer
{
    Real3 pos;
    Real3 dir;
};

//---------------------------------------------------------------------------//
/*!
 * View to a vector of VecGeom state information.
 *
 * This "view" is expected to be an argument to a geometry-related kernel
 * launch. It contains pointers to host-managed data.
 *
 * The \c vgstate and \c vgnext arguments must be the result of
 * vecgeom::NavStateContainer::GetGPUPointer; and they are only meaningful with
 * the corresponding \c vgmaxdepth, the result of \c GeoManager::getMaxDepth .
 */
struct GeoStatePointers
{
    size_type size       = 0;
    size_type vgmaxdepth = 0;
    void*     vgstate    = nullptr;
    void*     vgnext     = nullptr;

    Real3*     pos       = nullptr;
    Real3*     dir       = nullptr;
    real_type* next_step = nullptr;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return bool(size) && bool(vgmaxdepth) && bool(vgstate) && bool(vgnext)
               && bool(pos) && bool(dir) && bool(next_step);
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
