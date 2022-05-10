//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportData.cc
//---------------------------------------------------------------------------//
#include "ImportData.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the string value for a table type.
 */
const char* to_cstring(ImportEmParameter value)
{
    static const char* const strings[] = {"energy_loss_fluct",
                                          "lpm",
                                          "bins_per_decade",
                                          "min_table_energy",
                                          "max_table_energy"};

    CELER_EXPECT(static_cast<unsigned int>(value) * sizeof(const char*)
                 < sizeof(strings));
    return strings[static_cast<unsigned int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
