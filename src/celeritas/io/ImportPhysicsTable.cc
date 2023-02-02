//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsTable.cc
//---------------------------------------------------------------------------//
#include "ImportPhysicsTable.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
char const* to_cstring(ImportTableType value)
{
    static EnumStringMapper<ImportTableType> const to_cstring_impl{
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
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the string value for units.
 */
char const* to_cstring(ImportUnits value)
{
    static EnumStringMapper<ImportUnits> const to_cstring_impl{
        "unitless",
        "MeV",
        "MeV/cm",
        "cm",
        "1/cm",
        "1/cm-MeV",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
