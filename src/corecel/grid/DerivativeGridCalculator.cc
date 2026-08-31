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
 * Construct the derivative grid using interval slopes.
 *
 * Interior values are placed at the arithmetic mean of each interval's
 * endpoints. The first and last values use their nearest interval slope.
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
    CELER_VALIDATE(
        is_monotonic_increasing(make_span(grid.x)),
        << "input grid points must be strictly increasing for derivative "
           "calculation");

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

    // Interior points use the preceding interval slope at its midpoint
    for (size_type i = 1; i + 1 < n; ++i)
    {
        result.x[i] = real_type(0.5) * (grid.x[i - 1] + grid.x[i]);
        result.y[i] = derivative(i - 1);
    }

    // Last point uses the final interval
    result.y.back() = derivative(n - 2);

    CELER_ASSERT(result.x.size() == result.y.size());
    // Ensure epsilon neighborhoods don't overlap
    CELER_ASSERT(is_monotonic_nondecreasing(make_span(result.x)));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
