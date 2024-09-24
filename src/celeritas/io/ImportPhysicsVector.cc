//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsVector.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsVector.hh"

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Equality operator, mainly for debugging.
 */
bool operator==(ImportPhysics2DVector const& a, ImportPhysics2DVector const& b)
{
    return a.x == b.x && a.y == b.y && a.value == b.value;
}

//---------------------------------------------------------------------------//
/*!
 * Get the string value for a vector type.
 */
char const* to_cstring(ImportPhysicsVectorType value)
{
    static EnumStringMapper<ImportPhysicsVectorType> const to_cstring_impl{
        "unknown", "linear", "log", "free"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
