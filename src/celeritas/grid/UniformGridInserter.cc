//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGridInserter.cc
//---------------------------------------------------------------------------//
#include "UniformGridInserter.hh"

#include "corecel/Types.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
UniformGridInserter::UniformGridInserter(Values* reals, GridValues* grids)
    : reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
auto UniformGridInserter::operator()(UniformGridData const& grid,
                                     SpanConstDbl values) -> GridId
{
    CELER_EXPECT(grid);
    CELER_EXPECT(grid.size == values.size());

    UniformGridRecord data;
    data.grid = grid;
    data.value = reals_.insert_back(values.begin(), values.end());
    return grids_.push_back(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
