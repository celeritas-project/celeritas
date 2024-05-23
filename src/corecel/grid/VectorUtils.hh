//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/VectorUtils.hh
//! \brief Grid creation helpers
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Return evenly spaced numbers over a specific interval
std::vector<double> linspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
// Return logarithmically spaced numbers over a specific interval
std::vector<double> logspace(double start, double stop, size_type n);

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
}  // namespace celeritas
