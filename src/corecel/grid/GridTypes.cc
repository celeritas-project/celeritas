//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/GridTypes.cc
//---------------------------------------------------------------------------//
#include "GridTypes.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the interpolation method.
 */
char const* to_cstring(InterpolationType value)
{
    static EnumStringMapper<InterpolationType> const to_cstring_impl{
        "linear",
        "poly_spline",
        "cubic_spline",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
