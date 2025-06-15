//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/TwodGridBuilder.cc
//---------------------------------------------------------------------------//
#include "TwodGridBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to data that will be modified.
 */
TwodGridBuilder::TwodGridBuilder(Values* reals) : reals_{reals}
{
    CELER_EXPECT(reals);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid from an imported physics vector.
 */
TwodGridData TwodGridBuilder::operator()(inp::TwodGrid const& grid)
{
    CELER_EXPECT(grid);
    CELER_EXPECT(grid.x.size() >= 2 && grid.y.size() >= 2);

    TwodGridData result;
    result.x = reals_.insert_back(grid.x.begin(), grid.x.end());
    result.y = reals_.insert_back(grid.y.begin(), grid.y.end());
    result.values = reals_.insert_back(grid.value.begin(), grid.value.end());

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
