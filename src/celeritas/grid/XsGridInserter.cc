//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/XsGridInserter.cc
//---------------------------------------------------------------------------//
#include "XsGridInserter.hh"

#include "corecel/Types.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
XsGridInserter::XsGridInserter(Values* reals, GridValues* grids)
    : reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
auto XsGridInserter::operator()(inp::UniformGrid const& lower,
                                inp::UniformGrid const& upper) -> GridId
{
    CELER_EXPECT(lower || upper);

    XsGridRecord grid;
    if (lower)
    {
        grid.lower.grid = UniformGridData::from_bounds(lower.x, lower.y.size());
        grid.lower.value = reals_.insert_back(lower.y.begin(), lower.y.end());
    }
    if (upper)
    {
        grid.upper.grid = UniformGridData::from_bounds(upper.x, upper.y.size());
        grid.upper.value = reals_.insert_back(upper.y.begin(), upper.y.end());
    }
    CELER_ENSURE(grid);
    return grids_.push_back(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto XsGridInserter::operator()(inp::UniformGrid const& grid) -> GridId
{
    return (*this)(grid, {});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
