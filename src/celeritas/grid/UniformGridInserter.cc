//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGridInserter.cc
//---------------------------------------------------------------------------//
#include "UniformGridInserter.hh"

#include "corecel/Types.hh"
#include "corecel/grid/UniformGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
UniformGridInserter::UniformGridInserter(Values* reals, GridValues* grids)
    : values_(*reals), reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals && grids);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
auto UniformGridInserter::operator()(UniformGridData const& grid,
                                     SpanConstDbl values,
                                     bool spline) -> GridId
{
    return this->insert(grid, values, spline);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
auto UniformGridInserter::operator()(UniformGridData const& grid,
                                     SpanConstFlt values,
                                     bool spline) -> GridId
{
    return this->insert(grid, values, spline);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of physics data.
 */
template<class T>
auto UniformGridInserter::insert(UniformGridData const& grid,
                                 Span<T const> values,
                                 bool spline) -> GridId
{
    CELER_EXPECT(grid);
    CELER_EXPECT(grid.size == values.size());

    UniformGridRecord data;
    data.grid = grid;
    data.value = reals_.insert_back(values.begin(), values.end());
    if (spline)
    {
        // Calculate second derivatives for cubic spline interpolation
        ValuesRef ref(values_);
        auto deriv = SplineDerivCalculator(BC::geant)(data, ref);
        data.derivative = reals_.insert_back(deriv.begin(), deriv.end());
    }
    return grids_.push_back(data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
