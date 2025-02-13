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
auto XsGridInserter::operator()(UniformGridData const& lower_grid,
                                SpanConstDbl lower_values,
                                UniformGridData const& upper_grid,
                                SpanConstDbl upper_values) -> GridId
{
    return this->insert(lower_grid, lower_values, upper_grid, upper_values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
auto XsGridInserter::operator()(UniformGridData const& lower_grid,
                                SpanConstFlt lower_values,
                                UniformGridData const& upper_grid,
                                SpanConstFlt upper_values) -> GridId
{
    return this->insert(lower_grid, lower_values, upper_grid, upper_values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto XsGridInserter::operator()(UniformGridData const& grid,
                                SpanConstDbl values) -> GridId
{
    return (*this)(grid, values, UniformGridData{}, {});
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto XsGridInserter::operator()(UniformGridData const& grid,
                                SpanConstFlt values) -> GridId
{
    return (*this)(grid, values, UniformGridData{}, {});
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
template<class T>
auto XsGridInserter::insert(UniformGridData const& lower_grid,
                            Span<T const> lower_values,
                            UniformGridData const& upper_grid,
                            Span<T const> upper_values) -> GridId
{
    CELER_EXPECT(lower_grid || upper_grid);
    CELER_EXPECT(!lower_grid || lower_grid.size == lower_values.size());
    CELER_EXPECT(!upper_grid || upper_grid.size == upper_values.size());

    XsGridRecord grid;
    grid.lower.grid = lower_grid;
    grid.upper.grid = upper_grid;
    grid.lower.value
        = reals_.insert_back(lower_values.begin(), lower_values.end());
    grid.upper.value
        = reals_.insert_back(upper_values.begin(), upper_values.end());

    CELER_ENSURE(grid);
    return grids_.push_back(grid);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
