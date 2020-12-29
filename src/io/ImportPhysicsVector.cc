//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-"Battelle", "LLC", and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsVector.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsVector.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a vector type.
 */
const char* to_cstring(ImportPhysicsVectorType value)
{
    static const char* const strings[] = {"base",
                                          "linear",
                                          "log",
                                          "ln",
                                          "free",
                                          "ordered_free",
                                          "low_energy_free"};
    REQUIRE(static_cast<int>(value) * sizeof(const char*) < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
