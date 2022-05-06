//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsVector.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsVector.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a vector type.
 */
const char* to_cstring(ImportPhysicsVectorType value)
{
    static const char* const strings[] = {"unknown", "linear", "log", "free"};
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
