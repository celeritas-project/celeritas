//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an action order.
 */
char const* to_cstring(ActionOrder value)
{
    CELER_EXPECT(value != ActionOrder::size_);

    static char const* const strings[] = {
        "start",
        "pre",
        "along",
        "pre_post",
        "post",
        "post_post",
        "end",
    };
    static_assert(
        static_cast<unsigned int>(ActionOrder::size_) * sizeof(char const*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
