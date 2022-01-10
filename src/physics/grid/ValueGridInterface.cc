//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridInterface.cc
//---------------------------------------------------------------------------//
#include "ValueGridData.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
const char* to_cstring(ValueGridType value)
{
    static const char* const strings[] = {
        "macro_xs",
        "energy_loss",
        "range",
    };
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
