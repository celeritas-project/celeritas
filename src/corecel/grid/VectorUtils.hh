//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/VectorUtils.hh
//! \brief Grid creation helpers
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Return evenly spaced numbers over a specific interval
std::vector<double> linspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
// Return logarithmically spaced numbers over a specific interval
std::vector<double> geomspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
//! Return logarithmically spaced numbers over a specific interval
//! \deprecated Remove in v1.0; replaced by geomspace
[[deprecated]] inline std::vector<double>
logspace(double start, double stop, size_type n)
{
    return geomspace(start, stop, n);
}

//---------------------------------------------------------------------------//
/*!
 * True if the grid values are monotonically nondecreasing.
 */
template<class T>
inline bool is_monotonic_nondecreasing(Span<T> grid)
{
    return all_adjacent(grid.begin(), grid.end(), [](T& left, T& right) {
        return left <= right;
    });
}

//---------------------------------------------------------------------------//
/*!
 * True if the grid values are monotonically increasing.
 */
template<class T>
inline bool is_monotonic_increasing(Span<T> grid)
{
    return all_adjacent(grid.begin(), grid.end(), [](T& left, T& right) {
        return left < right;
    });
}

//---------------------------------------------------------------------------//
/*!
 * Calculate (geometric) ratio of successive grid points in a uniform log grid.
 */
template<class T>
T calc_log_delta(Span<T const> grid)
{
    CELER_EXPECT(grid.size() > 1);
    return fastpow(grid.back() / grid.front(), T(1) / (grid.size() - 1));
}

//---------------------------------------------------------------------------//
/*!
 * True if the grid has logarithmic spacing.
 */
template<class T>
inline bool has_log_spacing(Span<T const> grid)
{
    T delta = calc_log_delta(grid);
    for (auto i : range(grid.size() - 1))
    {
        if (!soft_equal(delta, grid[i + 1] / grid[i]))
            return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
