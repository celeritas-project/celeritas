//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_USE_SURF
#    include <VecGeom/surfaces/BrepHelper.h>
#endif

namespace celeritas
{
namespace detail
{
#ifdef VECGEOM_USE_SURF
//---------------------------------------------------------------------------//
// Set up surface tracking
void setup_surface_tracking_device(vgbrep::SurfData<vecgeom::Precision> const&);

// Tear down surface tracking
void teardown_surface_tracking_device();
#endif

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#ifndef VECGEOM_ENABLE_CUDA
#    ifdef VECGEOM_USE_SURF
// Set up surface tracking
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
