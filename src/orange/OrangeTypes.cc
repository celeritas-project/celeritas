//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.cc
//---------------------------------------------------------------------------//
#include "OrangeTypes.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a surface type.
 */
char const* to_cstring(SurfaceType value)
{
    static EnumStringMapper<SurfaceType> const to_cstring_impl
    {
        // clang-format off
        "px",
        "py",
        "pz",
        "cxc",
        "cyc",
        "czc",
        "sc",
#if 0
        "cx",
        "cy",
        "cz",
        "p",
#endif
        "s",
#if 0
        "kx",
        "ky",
        "kz",
        "sq",
#endif
        "gq",
        // clang-format on
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
