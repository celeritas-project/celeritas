//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.cc
//---------------------------------------------------------------------------//
#include "OrangeTypes.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a surface type.
 */
char const* to_cstring(SurfaceType value)
{
    CELER_EXPECT(value != SurfaceType::size_);

    static char const* const strings[] = {
        "px",
        "py",
        "pz",
        "cxc",
        "cyc",
        "czc",
        "sc",
#if 0
        "cx",
        "cy",
        "cz",
        "p",
#endif
        "s",
#if 0
        "kx",
        "ky",
        "kz",
        "sq",
#endif
        "gq",
    };
    static_assert(
        static_cast<unsigned int>(SurfaceType::size_) * sizeof(char const*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
