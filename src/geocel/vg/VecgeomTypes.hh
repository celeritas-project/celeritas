//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTypes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/base/Global.h>
#include <VecGeom/base/Version.h>
#include <VecGeom/navigation/NavStateFwd.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"

#if VECGEOM_VERSION < 0x020000 && CELERITAS_VECGEOM_SURFACE
#    error \
        "Unsupported: cannot build with VecGeom surface before merge into 2.0"
#endif

#ifndef VECGEOM_PRECISION_NAMESPACE
// VecGeom <= 2.0.0-rc.7 puts navindex, precision in global namespace
#    define VECGEOM_PRECISION_NAMESPACE
#endif

namespace vecgeom
{
VECGEOM_HOST_FORWARD_DECLARE(class LogicalVolume;);
VECGEOM_DEVICE_FORWARD_DECLARE(class LogicalVolume;);
VECGEOM_HOST_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_HOST_FORWARD_DECLARE(template<class T> class Vector3D;);
VECGEOM_DEVICE_FORWARD_DECLARE(template<class T> class Vector3D;);
}  // namespace vecgeom

namespace celeritas
{
//---------------------------------------------------------------------------//

using VgSurfaceInt = long;
using VgPlacedVolumeInt = int;
using vg_real_type = VECGEOM_PRECISION_NAMESPACE::Precision;

#if defined(VECGEOM_BVH_SINGLE) || defined(__DOXYGEN__)
using vgbvh_real_type = float;
#else
using vgbvh_real_type = double;
#endif

//---------------------------------------------------------------------------//
// LOW-LEVEL TYPES
//---------------------------------------------------------------------------//

//! VecGeom::VPlacedVolume::id is unsigned int
using VgPlacedVolumeId = OpaqueId<struct VecgeomPlacedVolume_,
                                  std::make_unsigned_t<VgPlacedVolumeInt>>;

//---------------------------------------------------------------------------//
// VOLUME/VECTOR TYPES
//---------------------------------------------------------------------------//

template<MemSpace M>
using VgLogicalVolume
    = MemSpaceCond_t<M, vecgeom::cxx::LogicalVolume, vecgeom::cuda::LogicalVolume>;

template<MemSpace M>
using VgPlacedVolume
    = MemSpaceCond_t<M, vecgeom::cxx::VPlacedVolume, vecgeom::cuda::VPlacedVolume>;

template<class T, MemSpace M>
using VgVector3
    = MemSpaceCond_t<M, vecgeom::cxx::Vector3D<T>, vecgeom::cuda::Vector3D<T>>;

using VgReal3 = VgVector3<vg_real_type, MemSpace::native>;

//---------------------------------------------------------------------------//
// NAVIGATION TYPES
//---------------------------------------------------------------------------//

//! High level VecGeom navigation state
using VgNavState = vecgeom::NavigationState;

//---------------------------------------------------------------------------//
}  // namespace celeritas
