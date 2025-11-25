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

#define CELER_VGNAV_TUPLE 1
#define CELER_VGNAV_INDEX 2
#define CELER_VGNAV_PATH 3

#if defined(VECGEOM_USE_NAVTUPLE)
#    define CELER_VGNAV CELER_VGNAV_TUPLE
#elif defined(VECGEOM_USE_NAVINDEX)
#    define CELER_VGNAV CELER_VGNAV_INDEX
#else
#    define CELER_VGNAV CELER_VGNAV_PATH
#endif

#ifdef VECGEOM_SINGLE_PRECISION
static_assert(CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT);
#else
static_assert(CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE);
#endif

namespace vecgeom
{
#if VECGEOM_VERSION >= 0x020000
template<VECGEOM_PRECISION_NAMESPACE::uint MD>
struct NavTuple;
#endif
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

#if VECGEOM_VERSION >= 0x020000
// Allow trivial copying of tuple between device/host
template<VECGEOM_PRECISION_NAMESPACE::uint MD>
struct TriviallyCopyable<vecgeom::NavTuple<MD>> : std::true_type
{
};
#endif

//---------------------------------------------------------------------------//
// LOW-LEVEL TYPES
//---------------------------------------------------------------------------//

//! VecGeom::VPlacedVolume::id is unsigned int
using VgVolumeInstanceId = OpaqueId<struct VecgeomPlacedVolume_,
                                    std::make_unsigned_t<VgPlacedVolumeInt>>;

enum class VgBoundary : bool
{
    off,
    on
};

CELER_CONSTEXPR_FUNCTION bool to_bool(VgBoundary b)
{
    return static_cast<bool>(b);
}

CELER_CONSTEXPR_FUNCTION VgBoundary to_vgboundary(bool b)
{
    return static_cast<VgBoundary>(b);
}

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

using VgNavIndex = VECGEOM_PRECISION_NAMESPACE::NavIndex_t;

//! Low-level (POD compatible) VecGeom navigation state
#if CELER_VGNAV == CELER_VGNAV_INDEX || defined(__DOXYGEN__)
using VgNavStateImpl = VgNavIndex;
#elif CELER_VGNAV == CELER_VGNAV_TUPLE
using VgNavStateImpl = vecgeom::NavTuple<VECGEOM_NAVTUPLE_MAXDEPTH>;
#elif CELER_VGNAV == CELER_VGNAV_PATH
// Only used clangd parsing of VgNavStateWrapper
using VgNavStateImpl = VgNavIndex;
#endif

//! High level VecGeom navigation state
using VgNavState = vecgeom::NavigationState;

//---------------------------------------------------------------------------//
}  // namespace celeritas
