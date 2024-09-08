//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantOpticalPhysicsOptions.cc
//---------------------------------------------------------------------------//
#include "GeantOpticalPhysicsOptions.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the wavelength shifting time model selection.
 */
char const* to_cstring(WLSTimeProfileSelection value)
{
    static EnumStringMapper<WLSTimeProfileSelection> const to_cstring_impl{
        "none",
        "delta",
        "exponential",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
