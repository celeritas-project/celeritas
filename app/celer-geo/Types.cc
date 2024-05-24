//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/sys/Device.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Convert a geometry enum to a string.
 */
char const* to_cstring(Geometry value)
{
    static EnumStringMapper<Geometry> const to_cstring_impl{
        "orange",
        "vecgeom",
        "geant4",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Default memory space for rendering.
 */
MemSpace default_memspace()
{
    return celeritas::device() ? MemSpace::device : MemSpace::host;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
