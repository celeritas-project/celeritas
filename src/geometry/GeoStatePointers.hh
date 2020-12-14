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

    //! Check whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        REQUIRE(this->valid());
        return bool(size);
    }

    // Whether the interface is valid
    inline CELER_FUNCTION bool valid() const;
};

struct GeoStateInitializer
{
    Real3 pos;
    Real3 dir;
};

//---------------------------------------------------------------------------//
// MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Check whether the state is consistently assigned.
 *
 * This is called as part of the bool operator, which should be checked as part
 * of an assertion immediately before launching a kernel and when returning a
 * state.
 */
CELER_FUNCTION bool GeoStatePointers::valid() const
{
    // clang-format off
    return    bool(size) == bool(vgmaxdepth)
           && bool(size) == bool(vgstate)
           && bool(size) == bool(vgnext)
           && bool(size) == bool(pos)
           && bool(size) == bool(dir)
           && bool(size) == bool(next_step);
    // clang-format on
}

//---------------------------------------------------------------------------//
} // namespace celeritas
