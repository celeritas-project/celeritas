//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/DerivativeGridCalculator.cc
//---------------------------------------------------------------------------//
#include "DerivativeGridCalculator.hh"

#include <fstream>

#include "GridTypes.hh"
#include "VectorUtils.hh"
#include "nlohmann/json.hpp"
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
    static std::ofstream out("derivative-mean.jsonl", std::ios::app);
    size_type const n = grid.x.size();
    result.x = grid.x;
    result.y.resize(n);

    auto derivative = [&](size_type i) {
        // CELER_EXPECT(i < grid.x.size());
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
