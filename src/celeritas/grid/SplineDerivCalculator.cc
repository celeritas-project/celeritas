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
 * Construct with x and y grids.
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
 * Construct with grid data.
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
    CELER_EXPECT(grid_->size() >= 5);

    if (bc_ == BoundaryCondition::not_not_a_knot)
    {
        // Calculate the second derivatives using the default Geant4 method
        // (which supposedly uses not-a-knot boundary conditions but produces
        // different results)
        return this->calc_geant_derivatives();
    }

    size_type num_knots = grid_->size();
    TridiagonalSolver::Coeffs coeffs(num_knots - 2);

    // Calculate the first row coefficients using the boundary conditions
    this->calc_initial_coeffs(coeffs[0]);

    // Calculate the interior row coefficients of the tridiagonal system
    for (size_type i = 2; i < num_knots - 2; ++i)
    {
        CELER_ASSERT(grid_->x(i) > grid_->x(i - 1));
        real_type h_lower = grid_->delta_x(i - 1);
        real_type h_upper = grid_->delta_x(i);

        coeffs[i - 1][0] = h_lower;
        coeffs[i - 1][1] = 2 * (h_lower + h_upper);
        coeffs[i - 1][2] = h_upper;
        coeffs[i - 1][3] = 6 * grid_->delta_slope(i);
    }

    // Calculate the last row coefficients using the boundary conditions
    this->calc_final_coeffs(coeffs[num_knots - 3]);

    // Solve the tridiagonal system
    VecReal result(num_knots);
    TridiagonalSolver(std::move(coeffs))({result.data() + 1, num_knots - 2});

    // Recover y''_0 and y''_{n - 1}
    this->calc_boundaries(result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients for the first row using the boundary conditions.
 */
void SplineDerivCalculator::calc_initial_coeffs(Real4& coeffs) const
{
    CELER_EXPECT(grid_->x(1) > grid_->x(0));

    real_type h_lower = grid_->delta_x(0);
    real_type h_upper = grid_->delta_x(1);

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
    coeffs[3] = 6 * grid_->delta_slope(1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients for the last row using the boundary conditions.
 */
void SplineDerivCalculator::calc_final_coeffs(Real4& coeffs) const
{
    CELER_EXPECT(grid_->x(grid_->size() - 2) > grid_->x(grid_->size() - 3));

    real_type h_lower = grid_->delta_x(grid_->size() - 3);
    real_type h_upper = grid_->delta_x(grid_->size() - 2);

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
    coeffs[3] = 6 * grid_->delta_slope(grid_->size() - 2);
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
        real_type h_lower = grid_->delta_x(0);
        real_type h_upper = grid_->delta_x(1);
        deriv.front() = ((h_lower + h_upper) * deriv[1] - h_lower * deriv[2])
                        / h_upper;

        h_lower = grid_->delta_x(grid_->size() - 3);
        h_upper = grid_->delta_x(grid_->size() - 2);
        deriv.back() = ((h_lower + h_upper) * deriv[grid_->size() - 2]
                        - h_upper * deriv[grid_->size() - 3])
                       / h_lower;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives.
 *
 * This is a hack to produce the same interpolation results as Geant4. The
 * calculation here is identical to Geant4's \c
 * G4PhysicsVector::ComputeSecDerivative1, which is based off the algorithm
 * for calculating the second derivatives of a cubic spline in
 * \cite{press-nr-1992}, modified for not-a-knot boundary conditions.
 *
 * Note that here the coefficients are divided by \f$ h_i + h_{i + 1} \f$.
 *
 * \todo While Geant4 supposedly uses not-a-knot boundary conditions, these
 * second derivatives differ from the expected values.
 */
auto SplineDerivCalculator::calc_geant_derivatives() const -> VecReal
{
    size_type num_knots = grid_->size();

    // Used to store the result as well as temporary storage for the decomposed
    // factors in the tridiagonal algorithm
    VecReal result(num_knots);
    VecReal rhs(num_knots - 1);

    // Set up the initial not-a-knot boundary conditions
    CELER_ASSERT(grid_->x(1) > grid_->x(0));
    real_type h_lower = grid_->delta_x(0);
    real_type h_upper = grid_->delta_x(1);

    // First \c c_prime value (negated) for the tridiagonal algorithm: -c' =
    // -a_2 / a_1.
    result[1] = (h_lower - h_upper) / (2 * h_upper + h_lower);

    // XXX Almost a_3 / a_1 (which would be 6 r_0 h_1 / ((h_0 + 2 h_1)(h_0 +
    // h_1)))
    rhs[1] = 6 * grid_->delta_slope(1) * h_upper / ipow<2>(h_lower + h_upper);

    // Tridiagonal algorithm decomposition and forward substitution
    for (size_type i = 2; i < num_knots - 2; ++i)
    {
        // Calculate the coefficients while performing the forward sweep
        h_lower = grid_->delta_x(i - 1);
        h_upper = grid_->delta_x(i);

        // a_0 = h_{i - 1} / (h_{i - 1} + h_i)
        real_type sig = h_lower / (h_lower + h_upper);

        // p = 1 / (a_1 - a_0 c'_{i - 1})
        real_type p = 1 / (2 + sig * result[i - 1]);

        // -c'_i = -a_2 p = h_{i} / ((h_{i - 1} + h_i) p)
        result[i] = (sig - 1) * p;

        // XXX Almost u_i = (a_3 - a_0 u_{i - 1}) p (note that the RHS a_3 is
        // not multiplied by p)
        rhs[i] = 6 * grid_->delta_slope(i) / (h_lower + h_upper)
                 - sig * rhs[i - 1] * p;
    }

    // Set up the final not-a-knot boundary conditions
    h_lower = grid_->delta_x(num_knots - 3);
    h_upper = grid_->delta_x(num_knots - 2);

    // XXX Calculate the next-to-last derivative outside of the back
    // substitution loop
    real_type sig = h_lower / (h_lower + h_upper);
    real_type p = 1 / (2 + sig * result[num_knots - 3]);
    rhs[num_knots - 2] = 6 * grid_->delta_slope(num_knots - 2) * sig
                             / (h_lower + h_upper)
                         - (2 * sig - 1) * rhs[num_knots - 3] * p;
    p = 1 / ((1 + sig) + (2 * sig - 1) * result[num_knots - 3]);
    result[num_knots - 2] = rhs[num_knots - 2] * p;

    // XXX Back substitution
    for (size_type i = num_knots - 3; i >= 1; --i)
    {
        h_lower = grid_->delta_x(i - 1);
        h_upper = grid_->delta_x(i);
        result[i] *= result[i + 1] - rhs[i] * (h_lower + h_upper) / h_upper;
    }

    // Recover y''_0 and y''_{n - 1}
    this->calc_boundaries(result);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
