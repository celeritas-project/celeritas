//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-"Battelle", "LLC", and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsTable.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsTable.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
const char* to_cstring(ImportTableType value)
{
    static const char* const strings[] = {
        "dedx",
        "dedx_subsec",
        "dedx_unrestricted",
        "ionisation",
        "ionisation_subsec",
        "csda_range",
        "range",
        "secondary_range",
        "inverse_range",
        "lambda",
        "sublambda",
        "lambda_prim",
    };
    CELER_EXPECT(static_cast<int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Get the string value for units.
 */
const char* to_cstring(ImportUnits value)
{
    static const char* const strings[] = {
        "unitless",
        "MeV",
        "MeV/cm",
        "cm",
        "1/cm",
        "1/cm-MeV",
    };
    CELER_EXPECT(static_cast<int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
