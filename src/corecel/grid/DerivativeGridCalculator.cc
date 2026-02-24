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
 * Construct with a given epsilon value.
 *
 * The provided epsilon value determines the size of the epsilon-neighborhood
 * around grid-points where the derivative might not be well-defined.
 */
DerivativeGridCalculator::DerivativeGridCalculator(real_type epsilon)
    : epsilon_(epsilon)
{
    CELER_EXPECT(epsilon > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the derivative grid of an imported grid.
 *
 * For each grid-point x in the input, the (x-epsilon, x+epsilon) is
 * constructed. Outside the neighborhood, the derivative is well defined,
 * whereas in the neighborhood it is interpolated between the endpoints of the
 * neighborhood.
 *
 * Since only linearly interpolated grids are supported, the derivative outside
 * the epsilon-neighborhood is just the interpolated slope between the
 * grid-points.
 */
inp::Grid DerivativeGridCalculator::operator()(inp::Grid const& grid)
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
    auto derivative = [&](size_type i) {
        if (i == 0 || i >= grid.x.size())
        {
            return real_type{0};
        }
        return (grid.y[i] - grid.y[i - 1]) / (grid.x[i] - grid.x[i - 1]);
    };

    for (size_type i : range(grid.x.size()))
    {
        // Add left-derivative grid-point
        result.x.push_back(grid.x[i] - epsilon_);
        result.y.push_back(derivative(i));

        // Add right-derivative grid-point
        result.x.push_back(grid.x[i] + epsilon_);
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
