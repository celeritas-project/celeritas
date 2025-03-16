//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomVersion.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Global.h>
#include <VecGeom/base/Math.h>
#include <VecGeom/base/Version.h>

#if VECGEOM_VERSION < 0x020000 && defined(VECGEOM_USE_SURF)
#    error \
        "Unsupported: cannot build with VecGeom surface before merge into 2.0"
#endif
#define VECGEOM_V2 0x020000

namespace vecgeom
{
//---------------------------------------------------------------------------//
#if VECGEOM_VERSION >= VECGEOM_V2
template<class T>
#endif
class ABBoxManager;

namespace cuda
{
#if VECGEOM_VERSION >= VECGEOM_V2
template<typename T>
#endif
class BVH;
}  // namespace cuda

//---------------------------------------------------------------------------//
}  // namespace vecgeom

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
#ifdef VECGEOM_BVH_SINGLE
using BvhPrecision = float;
#else
using BvhPrecision = double;
#endif

#if VECGEOM_VERSION >= VECGEOM_V2
using ABBoxManager_t = vecgeom::ABBoxManager<vecgeom::Precision>;
using BVH_t = vecgeom::cuda::BVH<BvhPrecision>;
#else
using ABBoxManager_t = vecgeom::ABBoxManager;
using BVH_t = vecgeom::BVH;
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
