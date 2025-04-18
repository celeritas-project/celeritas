//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGridInserter.cc
//---------------------------------------------------------------------------//
#include "UniformGridInserter.hh"

#include "corecel/Types.hh"
#include "corecel/grid/UniformGridData.hh"

#include "detail/GridUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
UniformGridInserter::UniformGridInserter(Values* reals, GridValues* grids)
    : values_(reals), reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
auto UniformGridInserter::operator()(inp::UniformGrid const& grid) -> GridId
{
    CELER_EXPECT(grid);

    UniformGridRecord data;
    data.grid = UniformGridData::from_bounds(grid.x, grid.y.size());
    data.value = reals_.insert_back(grid.y.begin(), grid.y.end());
    detail::set_spline(values_, reals_, grid.interpolation, data);

    CELER_ENSURE(data);
    return grids_.push_back(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
