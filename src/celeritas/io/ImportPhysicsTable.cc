//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
        "lambda",
        "lambda_prim",
        "dedx",
        "range",
        "msc_xs",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
