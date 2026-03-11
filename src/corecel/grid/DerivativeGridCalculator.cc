//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/DerivativeGridCalculator.cc
//---------------------------------------------------------------------------//
#include "DerivativeGridCalculator.hh"

#include "GridTypes.hh"
#include "VectorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the derivative grid of an imported grid.
 *
 * Since grid are piecewise functions, the left-derivatives and
 * right-derivatives might not agree at a grid point. Each x grid-point is
 * duplicated with the first value taking the left-derivative and the second
 * taking the right-derivative.
 *
 * \todo Currently only linearly interpolated grids are supported since they
 * are necessary for calculating group velocity from refractive index. The
 * endpoints of the input grid are assumed to be constant, and thus have 0
 * derivative.
 */
inp::Grid construct_derivative_grid(inp::Grid const& grid)
{
    CELER_EXPECT(grid);
    CELER_VALIDATE(grid.interpolation.type == InterpolationType::linear,
                   << to_cstring(grid.interpolation.type)
                   << " derivative calculation is not supported on a "
                      "non-linear grid");

    inp::Grid result;
    result.interpolation = grid.interpolation;

    result.x.reserve(2 * grid.x.size());
    result.y.reserve(2 * grid.y.size());

    // Calculate derivative for [i-1, i] grid interval
    auto derivative = [&](size_type i) -> double {
        if (i == 0 || i >= grid.x.size())
        {
            return 0;
        }
        return (grid.y[i] - grid.y[i - 1]) / (grid.x[i] - grid.x[i - 1]);
    };

    for (size_type i : range(grid.x.size()))
    {
        // Add left-derivative grid-point
        result.x.push_back(grid.x[i]);
        result.y.push_back(derivative(i));

        // Add right-derivative grid-point
        result.x.push_back(grid.x[i]);
        result.y.push_back(derivative(i + 1));
    }
    CELER_ASSERT(result.x.size() == 2 * grid.x.size());
    CELER_ASSERT(result.y.size() == 2 * grid.y.size());

    // Ensure epsilon neighborhoods don't overlap
    CELER_ASSERT(is_monotonic_nondecreasing(make_span(result.x)));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
