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
 */
struct VGView
{
    const vecgeom::VPlacedVolume* world_volume = nullptr;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGView_hh
