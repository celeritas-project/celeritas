//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridInserter.cc
//---------------------------------------------------------------------------//
#include "ValueGridInserter.hh"

#include "corecel/Types.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
ValueGridInserter::ValueGridInserter(Values* reals, GridValues* grids)
    : values_(reals), xs_grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
auto ValueGridInserter::operator()(UniformGridData const& log_grid,
                                   size_type prime_index,
                                   SpanConstDbl values) -> XsIndex
{
    CELER_EXPECT(log_grid);
    CELER_EXPECT(log_grid.size == values.size());
    CELER_EXPECT(prime_index <= log_grid.size
                 || prime_index == XsGridData::no_scaling());

    XsGridData grid;
    grid.log_energy = log_grid;
    grid.prime_index = prime_index;
    grid.value = values_.insert_back(values.begin(), values.end());
    return xs_grids_.push_back(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto ValueGridInserter::operator()(UniformGridData const& log_grid,
                                   SpanConstDbl values) -> XsIndex
{
    return (*this)(log_grid, XsGridData::no_scaling(), values);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
