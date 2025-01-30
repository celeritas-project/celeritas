//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineDerivCalculator.cc
//---------------------------------------------------------------------------//
#include "SplineDerivCalculator.hh"

#include "corecel/math/Algorithms.hh"
#include "corecel/math/TridiagonalSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Contruct with x and y grids.
 */
SplineDerivCalculator::SplineDerivCalculator(SpanConstReal x_values,
                                             SpanConstReal y_values)
    : grid_(std::make_unique<detail::SpanGridAccessor>(x_values, y_values))
{
}

//---------------------------------------------------------------------------//
/*!
 * Contruct with x and y grids.
 */
SplineDerivCalculator::SplineDerivCalculator(XsGridData const& grid,
                                             Values const& values)
    : grid_(std::make_unique<detail::XsGridAccessor>(grid, values))
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives.
 */
auto SplineDerivCalculator::operator()() const -> VecReal
{
    size_type num_knots = grid_->size();
    CELER_ASSERT(num_knots >= 5);

    TridiagonalSolver::Coeffs coeffs(num_knots - 2);

    CELER_ASSERT(grid_->x(1) > grid_->x(0));
    real_type dx_l = grid_->delta_x(0, 1);
    real_type dx_r = grid_->delta_x(1, 2);

    // Initial not-a-knot boundary conditions
    coeffs[0][0] = 0;
    coeffs[0][1] = (dx_l + dx_r) * (2 * dx_r + dx_l) / dx_r;
    coeffs[0][2] = (ipow<2>(dx_r) - ipow<2>(dx_l)) / dx_r;
    coeffs[0][3]
        = 6 * (grid_->delta_y(1, 2) / dx_r - grid_->delta_y(0, 1) / dx_l);

    // Fill the bands of the tridiagonal matrix and the RHS
    for (size_type i = 2; i < num_knots - 2; ++i)
    {
        CELER_ASSERT(grid_->x(i) > grid_->x(i - 1));
        dx_l = grid_->delta_x(i - 1, i);
        dx_r = grid_->delta_x(i, i + 1);

        coeffs[i - 1][0] = dx_l;
        coeffs[i - 1][1] = 2 * (dx_l + dx_r);
        coeffs[i - 1][2] = dx_r;
        coeffs[i - 1][3] = 6
                           * (grid_->delta_y(i, i + 1) / dx_r
                              - grid_->delta_y(i - 1, i) / dx_l);
    }

    size_type i = num_knots - 2;
    CELER_ASSERT(grid_->x(i) > grid_->x(i - 1));
    dx_l = grid_->delta_x(i - 1, i);
    dx_r = grid_->delta_x(i, i + 1);

    // Final not-a-knot boundary conditions
    coeffs[i - 1][0] = (ipow<2>(dx_l) - ipow<2>(dx_r)) / dx_l;
    coeffs[i - 1][1] = (dx_l + dx_r) * (2 * dx_l + dx_r) / dx_l;
    coeffs[i - 1][2] = 0;
    coeffs[i - 1][3] = 6
                       * (grid_->delta_y(i, i + 1) / dx_r
                          - grid_->delta_y(i - 1, i) / dx_l);

    // Solve the tridiagonal system
    VecReal result(num_knots);
    TridiagonalSolver(std::move(coeffs))({result.data() + 1, num_knots - 2});

    // Recover \f$ y''_0 \f$ and \f$ y''_n \f$
    result.front() = ((grid_->delta_x(0, 1) + grid_->delta_x(1, 2)) * result[1]
                      - grid_->delta_x(0, 1) * result[2])
                     / grid_->delta_x(1, 2);
    result.back() = ((dx_l + dx_r) * result[num_knots - 2]
                     - dx_r * result[num_knots - 3])
                    / dx_l;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives.
 */
/*
void SplineDerivCalculator::calc_interior()
{
}
*/

//---------------------------------------------------------------------------//
}  // namespace celeritas
