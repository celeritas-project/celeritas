//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportMaterial.cc
//---------------------------------------------------------------------------//
#include "ImportMaterial.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a printable label for material state.
 */
char const* to_cstring(ImportMaterialState value)
{
    static EnumStringMapper<ImportMaterialState> const to_cstring_impl{
        "other", "solid", "liquid", "gas"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a printable label for a scalar optical material property.
 */
char const* to_cstring(ImportOpticalScalar value)
{
    static EnumStringMapper<ImportOpticalScalar> const to_cstring_impl{
        "resolution_scale",
        "rise_time_fast",
        "rise_time_mid",
        "rise_time_slow",
        "fall_time_fast",
        "fall_time_mid",
        "fall_time_slow",
        "scint_yield",
        "scint_yield_fast",
        "scint_yield_mid",
        "scint_yield_slow",
        "lambda_mean_fast",
        "lambda_mean_mid",
        "lambda_mean_slow",
        "lambda_sigma_fast",
        "lambda_sigma_mid",
        "lambda_sigma_slow"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a printable label for a vector optical material property.
 */
char const* to_cstring(ImportOpticalVector value)
{
    static EnumStringMapper<ImportOpticalVector> const to_cstring_impl{
        "refractive_index"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
