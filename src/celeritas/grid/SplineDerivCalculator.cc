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
                                             SpanConstReal y_values,
                                             BoundaryCondition bc)
    : grid_(std::make_unique<detail::SpanGridAccessor>(x_values, y_values))
    , bc_(bc)
{
    CELER_EXPECT(bc_ != BoundaryCondition::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Contruct with grid data.
 */
SplineDerivCalculator::SplineDerivCalculator(XsGridData const& grid,
                                             Values const& values,
                                             BoundaryCondition bc)
    : grid_(std::make_unique<detail::XsGridAccessor>(grid, values)), bc_(bc)
{
    CELER_EXPECT(bc_ != BoundaryCondition::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives.
 */
auto SplineDerivCalculator::operator()() const -> VecReal
{
    CELER_ASSERT(grid_->size() >= 5);

    if (bc_ == BoundaryCondition::geant)
    {
        // return this->calc_geant_derivatives();
    }

    size_type num_knots = grid_->size();
    TridiagonalSolver::Coeffs coeffs(num_knots - 2);

    // Calculate the first row coefficients using the boundary conditions
    this->calc_initial_row(coeffs[0]);

    // Calculate the interior rows of the tridiagonal matrix and the RHS
    for (size_type i = 2; i < num_knots - 2; ++i)
    {
        CELER_ASSERT(grid_->x(i) > grid_->x(i - 1));
        real_type h_lower = grid_->delta_x(i - 1, i);
        real_type h_upper = grid_->delta_x(i, i + 1);

        coeffs[i - 1][0] = h_lower;
        coeffs[i - 1][1] = 2 * (h_lower + h_upper);
        coeffs[i - 1][2] = h_upper;
        coeffs[i - 1][3] = 6
                           * (grid_->delta_y(i, i + 1) / h_upper
                              - grid_->delta_y(i - 1, i) / h_lower);
    }

    // Calculate the last row coefficients using the boundary conditions
    this->calc_final_row(coeffs[num_knots - 3]);

    // Solve the tridiagonal system
    VecReal result(num_knots);
    TridiagonalSolver(std::move(coeffs))({result.data() + 1, num_knots - 2});

    // Recover \f$ y''_0 \f$ and \f$ y''_n \f$
    this->calc_boundaries(result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients for the first row using the boundary conditions.
 */
void SplineDerivCalculator::calc_initial_row(Real4& coeffs) const
{
    CELER_EXPECT(grid_->x(1) > grid_->x(0));

    real_type h_lower = grid_->delta_x(0, 1);
    real_type h_upper = grid_->delta_x(1, 2);

    coeffs[0] = 0;
    if (bc_ == BoundaryCondition::natural)
    {
        coeffs[1] = 2 * (h_lower + h_upper);
        coeffs[2] = h_upper;
    }
    else
    {
        coeffs[1] = (h_lower + h_upper) * (2 * h_upper + h_lower) / h_upper;
        coeffs[2] = (ipow<2>(h_upper) - ipow<2>(h_lower)) / h_upper;
    }
    coeffs[3]
        = 6 * (grid_->delta_y(1, 2) / h_upper - grid_->delta_y(0, 1) / h_lower);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients for the last row using the boundary conditions.
 */
void SplineDerivCalculator::calc_final_row(Real4& coeffs) const
{
    CELER_EXPECT(grid_->x(grid_->size() - 2) > grid_->x(grid_->size() - 3));

    size_type n = grid_->size() - 2;
    real_type h_lower = grid_->delta_x(n - 1, n);
    real_type h_upper = grid_->delta_x(n, n + 1);

    if (bc_ == BoundaryCondition::natural)
    {
        coeffs[0] = h_lower;
        coeffs[1] = 2 * (h_lower + h_upper);
    }
    else
    {
        coeffs[0] = (ipow<2>(h_lower) - ipow<2>(h_upper)) / h_lower;
        coeffs[1] = (h_lower + h_upper) * (2 * h_lower + h_upper) / h_lower;
    }
    coeffs[2] = 0;
    coeffs[3] = 6
                * (grid_->delta_y(n, n + 1) / h_upper
                   - grid_->delta_y(n - 1, n) / h_lower);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the first and last values of the second derivative.
 */
void SplineDerivCalculator::calc_boundaries(VecReal& deriv) const
{
    CELER_EXPECT(deriv.size() == grid_->size());

    if (bc_ == BoundaryCondition::natural)
    {
        deriv.front() = 0;
        deriv.back() = 0;
    }
    else
    {
        real_type h_lower = grid_->delta_x(0, 1);
        real_type h_upper = grid_->delta_x(1, 2);

        deriv.front() = ((h_lower + h_upper) * deriv[1] - h_lower * deriv[2])
                        / h_upper;

        size_type n = grid_->size() - 2;
        h_lower = grid_->delta_x(n - 1, n);
        h_upper = grid_->delta_x(n, n + 1);

        deriv.back() = ((h_lower + h_upper) * deriv[n] - h_upper * deriv[n - 1])
                       / h_lower;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
