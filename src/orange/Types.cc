//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.cc
//---------------------------------------------------------------------------//
#include "Types.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a surface type.
 */
const char* to_cstring(SurfaceType value)
{
    CELER_EXPECT(value != SurfaceType::size_);

    static const char* const strings[] = {
        "px",
        "py",
        "pz",
        "cxc",
        "cyc",
        "czc",
#if 0
        "sc",
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
        static_cast<unsigned int>(SurfaceType::size_) * sizeof(const char*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
