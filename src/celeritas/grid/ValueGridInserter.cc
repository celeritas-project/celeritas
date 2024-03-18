//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
ValueGridInserter::ValueGridInserter(Items<real_type>* reals,
                                     Items<XsGridData>* xs)
    : values_(reals), xs_grids_(xs)
{
    CELER_EXPECT(reals && xs);
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
/*!
 * Add a grid of host pointer data.
 */
auto ValueGridInserter::operator()(InterpolatedGrid, InterpolatedGrid)
    -> GenericIndex
{
    CELER_NOT_IMPLEMENTED("generic grids");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
