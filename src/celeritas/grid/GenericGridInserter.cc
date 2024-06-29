//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridInserter.cc
//---------------------------------------------------------------------------//
#include "GenericGridInserter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with collections to be populated.
 */
GenericGridSingleInserter::GenericGridSingleInserter(RealCollection* reals)
    : reals_(reals)
{
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
GenericGridData
GenericGridSingleInserter::operator()(SpanConstFlt grid, SpanConstFlt values)
{
    return this->insert_impl(grid, values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 */
GenericGridData
GenericGridSingleInserter::operator()(SpanConstDbl grid, SpanConstDbl values)
{
    return this->insert_impl(grid, values);
}

//---------------------------------------------------------------------------//
/*!
 * Add an imported physics vector as a generic grid.
 */
GenericGridData
GenericGridSingleInserter::operator()(ImportPhysicsVector const& vec)
{
    return this->insert_impl(make_span(vec.x), make_span(vec.y));
}

//---------------------------------------------------------------------------//
/*!
 * Add an imported physics vector as a generic grid.
 */
GenericGridData GenericGridSingleInserter::operator()()
{
    return GenericGridData{};
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of generic data with linear interpolation.
 *
 * This is templated to support real_type being single or double precision.
 */
template<class T>
GenericGridData
GenericGridSingleInserter::insert_impl(Span<T const> grid, Span<T const> values)
{
    CELER_EXPECT(grid.size() >= 2);
    CELER_EXPECT(grid.front() <= grid.back());
    CELER_EXPECT(values.size() == grid.size());

    GenericGridData grid_data;
    grid_data.grid = reals_.insert_back(grid.begin(), grid.end());
    grid_data.value = reals_.insert_back(values.begin(), values.end());

    CELER_ENSURE(grid_data);
    return grid_data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
