//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/FindInterp.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
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
    size_type index{};  //!< Lower index into the grid
    T fraction{};  //!< Fraction of the value between its neighbors
};

template<class T>
CELER_FUNCTION FindInterp(size_type, T) -> FindInterp<T>;

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
 * one. If the requested point is exactly on a coincident grid point, the lower
 * point and a fraction of zero will result.
 */
template<class Grid>
inline CELER_FUNCTION auto
find_interp(Grid const& grid, typename Grid::value_type value)
{
    CELER_EXPECT(value >= grid.front() && value < grid.back());

    auto index = grid.find(value);
    CELER_ASSERT(index + 1 < grid.size());
    auto const lower_val = grid[index];
    auto const upper_val = grid[index + 1];
    return FindInterp{index, (value - lower_val) / (upper_val - lower_val)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
