//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsTable.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsTable.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
char const* to_cstring(ImportTableType value)
{
    static char const* const strings[] = {
        "dedx",
        "dedx_process",
        "dedx_subsec",
        "dedx_unrestricted",
        "ionization",
        "ionization_subsec",
        "csda_range",
        "range",
        "secondary_range",
        "inverse_range",
        "lambda",
        "sublambda",
        "lambda_prim",
    };
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(char const*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Get the string value for units.
 */
char const* to_cstring(ImportUnits value)
{
    static char const* const strings[] = {
        "unitless",
        "MeV",
        "MeV/cm",
        "cm",
        "1/cm",
        "1/cm-MeV",
    };
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(char const*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
