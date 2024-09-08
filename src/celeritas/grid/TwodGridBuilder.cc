//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
TwodGridBuilder::TwodGridBuilder(Items<real_type>* reals) : reals_{reals}
{
    CELER_EXPECT(reals);
}

//---------------------------------------------------------------------------//
/*!
 * Add a 2D grid of float data.
 */
auto TwodGridBuilder::operator()(SpanConstFlt grid_x,
                                 SpanConstFlt grid_y,
                                 SpanConstFlt values) -> TwodGrid
{
    return this->insert_impl(grid_x, grid_y, values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a 2D grid of double data.
 */
auto TwodGridBuilder::operator()(SpanConstDbl grid_x,
                                 SpanConstDbl grid_y,
                                 SpanConstDbl values) -> TwodGrid
{
    return this->insert_impl(grid_x, grid_y, values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a 2D grid from container references.
 */
template<class T>
auto TwodGridBuilder::insert_impl(Span<T const> grid_x,
                                  Span<T const> grid_y,
                                  Span<T const> values) -> TwodGrid
{
    CELER_EXPECT(grid_x.size() >= 2);
    CELER_EXPECT(grid_x.front() <= grid_x.back());
    CELER_EXPECT(grid_y.size() >= 2);
    CELER_EXPECT(grid_y.front() <= grid_y.back());
    CELER_EXPECT(values.size() == grid_x.size() * grid_y.size());

    TwodGrid result;
    result.x = reals_.insert_back(grid_x.begin(), grid_x.end());
    result.y = reals_.insert_back(grid_y.begin(), grid_y.end());
    result.values = reals_.insert_back(values.begin(), values.end());

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
