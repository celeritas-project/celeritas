//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridType.cc
//---------------------------------------------------------------------------//
#include "ValueGridType.hh"

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string representation of a grid.
 */
char const* to_cstring(ValueGridType value)
{
    static EnumStringMapper<ValueGridType> const to_cstring_impl{
        "macro_xs", "energy_loss", "range"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
