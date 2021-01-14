//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoStatePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Types.hh"

namespace celeritas
{
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

//! Data required to initialize a geometry state
struct GeoStateInitializer
{
    Real3 pos;
    Real3 dir;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
