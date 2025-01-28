//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineDerivativeCalculator.cc
//---------------------------------------------------------------------------//
#include "SplineDerivativeCalculator.hh"

#include "corecel/math/Algorithms.hh"
#include "corecel/math/TridiagonalSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Contruct with x and y grids.
 */
SplineDerivativeCalculator::SplineDerivativeCalculator(SpanConstReal x_values,
                                                       SpanConstReal y_values)
    : grid_(std::make_unique<detail::SpanGridAccessor>(x_values, y_values))
{
}

//---------------------------------------------------------------------------//
/*!
 * Contruct with x and y grids.
 */
SplineDerivativeCalculator::SplineDerivativeCalculator(XsGridData const& grid,
                                                       Values const& values)
    : grid_(std::make_unique<detail::XsGridAccessor>(grid, values))
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the first derivatives.
 */
auto SplineDerivativeCalculator::operator()() const -> VecReal
{
    size_type num_points = grid_->size();
    CELER_ASSERT(num_points >= 3);

    VecReal result(num_points);
    TridiagonalSolver::Coefficients coeffs;
    resize(coeffs, num_points);

    real_type h_lower = grid_->delta_x(0);
    real_type h_upper = grid_->delta_x(1);
    real_type r_lower = (grid_->delta_y(0)) / h_lower;
    real_type r_upper = (grid_->delta_y(1)) / h_upper;

    if (num_points == 3)
    {
        // Handle the special case with three points where both conditions are
        // idential with not-a-knot boundaries by constructing a parabola
        // throught the points
        coeffs.a = {0, h_upper, 1};
        coeffs.b = {1, 2 * (h_lower + h_upper), h_lower};
        coeffs.c = {1, h_lower, 0};
        coeffs.d = {2 * r_lower,
                    3 * (h_lower * r_upper + h_upper * r_lower),
                    2 * r_upper};
        return TridiagonalSolver(std::move(coeffs))();
    }

    // Not-a-knot boundary conditions
    coeffs.a[0] = 0;
    coeffs.b[0] = h_upper * (h_lower + h_upper);
    coeffs.c[0] = ipow<2>(h_lower + h_upper);
    coeffs.d[0] = r_lower * (3 * h_lower * h_upper + 2 * ipow<2>(h_upper))
                  + r_upper * ipow<2>(h_lower);

    // Fill RHS and bands of tridiagonal matrix
    for (size_type i = 1; i < num_points - 1; ++i)
    {
        h_lower = grid_->delta_x(i - 1);
        h_upper = grid_->delta_x(i);
        r_lower = (grid_->delta_y(i - 1)) / h_lower;
        r_upper = (grid_->delta_y(i)) / h_upper;

        coeffs.a[i] = h_upper;
        coeffs.b[i] = 2 * (h_lower + h_upper);
        coeffs.c[i] = h_lower;
        coeffs.d[i] = 3 * (r_lower * h_upper + r_upper * h_lower);
    }

    // Not-a-knot boundary conditions
    coeffs.a[num_points - 1] = ipow<2>(h_lower + h_upper);
    coeffs.b[num_points - 1] = h_lower * (h_lower + h_upper);
    coeffs.c[num_points - 1] = 0;
    coeffs.d[num_points - 1]
        = r_lower * ipow<2>(h_upper)
          + r_upper * (3 * h_lower * h_upper + 2 * ipow<2>(h_lower));

    return TridiagonalSolver(std::move(coeffs))();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
