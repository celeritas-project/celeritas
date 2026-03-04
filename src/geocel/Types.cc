//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a geometry track state.
 */
char const* to_cstring(GeoStatus value)
{
    static EnumStringMapper<GeoStatus> const to_cstring_impl{
        "interior",
        "exiting_boundary",
        "entering_boundary",
        "error",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
