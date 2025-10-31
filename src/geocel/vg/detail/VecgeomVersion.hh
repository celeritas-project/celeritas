//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomVersion.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <VecGeom/base/BVH.h>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/base/Global.h>
#include <VecGeom/base/Math.h>
#include <VecGeom/base/Version.h>
#include <VecGeom/management/ABBoxManager.h>

#if VECGEOM_VERSION < 0x020000 && CELERITAS_VECGEOM_SURFACE
#    error \
        "Unsupported: cannot build with VecGeom surface before merge into 2.0"
#endif

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

using NavIndex_t = ::NavIndex_t;

#if VECGEOM_VERSION >= 0x020000
using ABBoxManager_t = vecgeom::ABBoxManager<vecgeom::Precision>;
using CudaBVH_t = vecgeom::cuda::BVH<BvhPrecision>;
#else
using ABBoxManager_t = vecgeom::ABBoxManager;
using CudaBVH_t = vecgeom::cuda::BVH;
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
