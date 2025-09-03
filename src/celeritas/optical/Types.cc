//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an optical surface physics step.
 */
char const* to_cstring(SurfacePhysicsOrder step)
{
    static EnumStringMapper<SurfacePhysicsOrder> const to_cstring_impl{
        "roughness",
        "reflectivity",
        "interaction",
    };
    return to_cstring_impl(step);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
