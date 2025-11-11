//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomTypes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Version.h>

#if VECGEOM_VERSION < 0x020000 && CELERITAS_VECGEOM_SURFACE
#    error \
        "Unsupported: cannot build with VecGeom surface before merge into 2.0"
#endif

#ifndef VECGEOM_PRECISION_NAMESPACE
// VecGeom <= 2.0.0-rc.7 puts navindex, precision in global namespace
#    define VECGEOM_PRECISION_NAMESPACE
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

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
