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
}  // namespace celeritas
