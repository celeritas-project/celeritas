//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.cc
//---------------------------------------------------------------------------//
#include "RunInput.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list selection.
 */
char const* to_cstring(PhysicsListSelection value)
{
    static EnumStringMapper<PhysicsListSelection> const to_cstring_impl{
        "ftfp_bert",
        "geant_physics_list",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
