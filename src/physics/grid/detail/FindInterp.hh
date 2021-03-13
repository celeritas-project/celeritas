//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GridInterp.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Result of finding a point on a grid for interpolating.
 *
 * The resulting index will be in [0, grid.size() - 1)
 * and the fraction will be in [0, 1).
 */
template<class T>
struct FindInterp
{
    size_type index{};    //!< Lower index into the grid
    T         fraction{}; //!< Fraction of the value between its neighbors
};

//---------------------------------------------------------------------------//
/*!
 * Find the index of the value and its fraction between neighboring points.
 *
 * The grid class should have a floating point value and must have methods \c
 * find, \c front, \c back, and \c operator[] .
 *
 * The value must be bounded by the grid and less than the final value. The
 * result will always have an index such that its neighbor to the right is a
 * valid point on the grid, and the fraction between neghbors may be zero (in
 * the case where the value is exactly on a grid point) but is always less than
 * one.
 */
template<class Grid>
inline CELER_FUNCTION FindInterp<typename Grid::value_type>
find_interp(const Grid& grid, typename Grid::value_type value)
{
    CELER_EXPECT(value >= grid.front() && value < grid.back());

    FindInterp<typename Grid::value_type> result;
    result.index = grid.find(value);
    CELER_ASSERT(result.index + 1 < grid.size());
    const auto lower_val = grid[result.index];
    const auto upper_val = grid[result.index + 1];
    result.fraction      = (value - lower_val) / (upper_val - lower_val);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
