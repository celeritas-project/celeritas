//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridInterface.cc
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"

#include "ValueGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
char const* to_cstring(ValueGridType value)
{
    static char const* const strings[]
        = {"macro_xs", "energy_loss", "range", "msc_mfp"};
    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(char const*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
