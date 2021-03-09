//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridInserter.cc
//---------------------------------------------------------------------------//
#include "ValueGridInserter.hh"

#include "base/SpanRemapper.hh"
#include "base/VectorUtils.hh"
#include "comm/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
ValueGridInserter::ValueGridInserter(RealCollection*   real_data,
                                     XsGridCollection* xs_grid)
    : values_(real_data), xs_grids_(xs_grid)
{
    CELER_EXPECT(real_data && xs_grid);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics xs data.
 */
auto ValueGridInserter::operator()(const UniformGridData& log_grid,
                                   size_type              prime_index,
                                   SpanConstReal          values) -> XsIndex
{
    CELER_EXPECT(log_grid);
    CELER_EXPECT(log_grid.size == values.size());
    CELER_EXPECT(prime_index <= log_grid.size
                 || prime_index == XsGridData::no_scaling());

    XsGridData grid;
    grid.log_energy  = log_grid;
    grid.prime_index = prime_index;
    grid.value       = values_.insert_back(values.begin(), values.end());
    return xs_grids_.push_back(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of log-spaced data without 1/E scaling.
 */
auto ValueGridInserter::operator()(const UniformGridData& log_grid,
                                   SpanConstReal          values) -> XsIndex
{
    return (*this)(log_grid, XsGridData::no_scaling(), values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of host pointer data.
 */
auto ValueGridInserter::operator()(InterpolatedGrid, InterpolatedGrid)
    -> GenericIndex
{
    CELER_NOT_IMPLEMENTED("generic grids");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
