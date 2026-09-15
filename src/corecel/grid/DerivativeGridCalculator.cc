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
 * right-derivatives might not agree at a grid point. A single derivative
 * is stored at each point by taking the harmonic mean of the adjacent interval
 * slopes. This reduces the influence of a comparatively
 * large slope while incorporating both one-sided derivatives.
 * The endpoints use the one-sided slopes of their nearest intervals.
 *
 * \todo Currently only linearly interpolated grids are supported since they
 * are necessary for calculating group velocity from refractive index.
 */
inp::Grid construct_derivative_grid(inp::Grid const& grid)
{
    CELER_EXPECT(grid);
    CELER_VALIDATE(grid.interpolation.type == InterpolationType::linear,
                   << to_cstring(grid.interpolation.type)
                   << " derivative calculation is not supported on a "
                      "non-linear grid");
    CELER_VALIDATE(is_monotonic_increasing(make_span(grid.x)),
                   << "input grid has coincident x values; harmonic-mean "
                      "derivative is undefined across a zero-width interval");

    inp::Grid result;
    result.interpolation = grid.interpolation;

    size_type const n = grid.x.size();
    result.x = grid.x;
    result.y.resize(n);

    auto derivative = [&](size_type i) {
        CELER_EXPECT(i + 1 < grid.x.size());
        return (grid.y[i + 1] - grid.y[i]) / (grid.x[i + 1] - grid.x[i]);
    };

    // One-sided derivatives at endpoints
    result.y.front() = derivative(0);
    result.y.back() = derivative(n - 2);
    for (size_type i = 1; i + 1 < n; ++i)
    {
        double const s_left = derivative(i - 1);
        double const s_right = derivative(i);

        result.y[i] = 2 * s_left * s_right / (s_left + s_right);
    }

    CELER_ASSERT(result.x.size() == result.y.size());
    // Ensure epsilon neighborhoods don't overlap
    CELER_ASSERT(is_monotonic_nondecreasing(make_span(result.x)));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
