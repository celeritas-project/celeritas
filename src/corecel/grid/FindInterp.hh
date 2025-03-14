//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/FindInterp.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/detail/TypeTraits.hh"
#include "corecel/math/Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Result of finding a point on a grid for interpolating.
 *
 * The resulting index will be in [0, grid.size() - 1)
 * and the fraction will be in [0, 1).
 */
struct FindInterp
{
    size_type index{};  //!< Lower index into the grid
    real_type fraction{};  //!< Fraction of the value between its neighbors
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
inline CELER_FUNCTION FindInterp find_interp(Grid const& grid,
                                             typename Grid::value_type value)
{
    CELER_EXPECT(value >= grid.front() && value < grid.back());

    FindInterp result;
    result.index = grid.find(value);
    CELER_ASSERT(result.index + 1 < grid.size());
    auto const lower_val = grid[result.index];
    auto const upper_val = grid[result.index + 1];
    using value_type = typename Grid::value_type;
    if constexpr (detail::is_quantity_v<value_type>)
    {
        result.fraction = value_as<value_type>(value - lower_val)
                          / value_as<value_type>(upper_val - lower_val);
    }
    else
    {
        result.fraction = (value - lower_val) / (upper_val - lower_val);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
