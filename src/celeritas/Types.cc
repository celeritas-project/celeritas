//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a state of matter.
 */
char const* to_cstring(MatterState value)
{
    static EnumStringMapper<MatterState> const to_cstring_impl{
        "unspecified",
        "solid",
        "liquid",
        "gas",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an action order.
 */
char const* to_cstring(ActionOrder value)
{
    static EnumStringMapper<ActionOrder> const to_cstring_impl{
        "start",
        "sort_start",
        "pre",
        "sort_pre",
        "along",
        "pre_post",
        "post",
        "post_post",
        "end",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
