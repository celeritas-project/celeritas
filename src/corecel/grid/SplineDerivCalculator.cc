//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/SplineDerivCalculator.cc
//---------------------------------------------------------------------------//
#include "SplineDerivCalculator.hh"

#include "corecel/math/Algorithms.hh"
#include "corecel/math/TridiagonalSolver.hh"

#include "detail/GridAccessor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from boundary conditions.
 */
SplineDerivCalculator::SplineDerivCalculator(BoundaryCondition bc) : bc_(bc)
{
    CELER_EXPECT(bc_ != BoundaryCondition::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives from grid data.
 */
auto SplineDerivCalculator::operator()(NonuniformGridRecord const& data,
                                       Values const& reals) const -> VecReal
{
    return (*this)(detail::NonuniformGridAccessor(data, reals));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives from grid data.
 */
auto SplineDerivCalculator::operator()(UniformGridRecord const& data,
                                       Values const& reals) const -> VecReal
{
    return (*this)(detail::UniformGridAccessor(data, reals));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives from spans.
 */
auto SplineDerivCalculator::operator()(SpanConstReal x, SpanConstReal y) const
    -> VecReal
{
    return (*this)(detail::NonuniformGridAccessor(x, y));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives.
 */
template<class GA>
auto SplineDerivCalculator::operator()(GA&& grid) const -> VecReal
{
    CELER_EXPECT(grid.size() >= min_grid_size());

    if (bc_ == BoundaryCondition::geant)
    {
        // Calculate the second derivatives using the default Geant4 method
        // (which supposedly uses not-a-knot boundary conditions but produces
        // different results)
        return this->calc_geant_derivatives(grid);
    }

    size_type num_knots = grid.size();
    TridiagonalSolver::Coeffs tridiag(num_knots - 2);
    VecReal rhs(num_knots - 2);

    // Calculate the first row coefficients using the boundary conditions
    this->calc_initial_coeffs(grid, tridiag[0], rhs[0]);

    // Calculate the interior row coefficients of the tridiagonal system
    for (size_type i = 2; i < num_knots - 2; ++i)
    {
        real_type h_lower = grid.delta_x(i - 1);
        real_type h_upper = grid.delta_x(i);

        tridiag[i - 1][0] = h_lower;
        tridiag[i - 1][1] = 2 * (h_lower + h_upper);
        tridiag[i - 1][2] = h_upper;
        rhs[i - 1] = 6 * grid.delta_slope(i);
    }

    // Calculate the last row coefficients using the boundary conditions
    this->calc_final_coeffs(grid, tridiag[num_knots - 3], rhs[num_knots - 3]);

    // Solve the tridiagonal system
    VecReal result(num_knots);
    TridiagonalSolver(std::move(tridiag))(
        make_span(rhs), make_span(result).subspan(1, num_knots - 2));

    // Recover \f$ S''_0 \f$ and \f$ S''_{n - 1} \f$
    this->calc_boundaries(grid, result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients for the first row using the boundary conditions.
 */
template<class GA>
void SplineDerivCalculator::calc_initial_coeffs(GA const& grid,
                                                Real3& tridiag,
                                                real_type& rhs) const
{
    real_type h_lower = grid.delta_x(0);
    real_type h_upper = grid.delta_x(1);

    tridiag[0] = 0;
    if (bc_ == BoundaryCondition::natural)
    {
        tridiag[1] = 2 * (h_lower + h_upper);
        tridiag[2] = h_upper;
    }
    else
    {
        tridiag[1] = (h_lower + h_upper) * (2 * h_upper + h_lower) / h_upper;
        tridiag[2] = (ipow<2>(h_upper) - ipow<2>(h_lower)) / h_upper;
    }
    rhs = 6 * grid.delta_slope(1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the coefficients for the last row using the boundary conditions.
 */
template<class GA>
void SplineDerivCalculator::calc_final_coeffs(GA const& grid,
                                              Real3& tridiag,
                                              real_type& rhs) const
{
    real_type h_lower = grid.delta_x(grid.size() - 3);
    real_type h_upper = grid.delta_x(grid.size() - 2);

    if (bc_ == BoundaryCondition::natural)
    {
        tridiag[0] = h_lower;
        tridiag[1] = 2 * (h_lower + h_upper);
    }
    else
    {
        tridiag[0] = (ipow<2>(h_lower) - ipow<2>(h_upper)) / h_lower;
        tridiag[1] = (h_lower + h_upper) * (2 * h_lower + h_upper) / h_lower;
    }
    tridiag[2] = 0;
    rhs = 6 * grid.delta_slope(grid.size() - 2);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the first and last values of the second derivative.
 */
template<class GA>
void SplineDerivCalculator::calc_boundaries(GA const& grid, VecReal& deriv) const
{
    CELER_EXPECT(deriv.size() == grid.size());

    if (bc_ == BoundaryCondition::natural)
    {
        deriv.front() = 0;
        deriv.back() = 0;
    }
    else
    {
        real_type h_lower = grid.delta_x(0);
        real_type h_upper = grid.delta_x(1);
        deriv.front() = ((h_lower + h_upper) * deriv[1] - h_lower * deriv[2])
                        / h_upper;

        h_lower = grid.delta_x(grid.size() - 3);
        h_upper = grid.delta_x(grid.size() - 2);
        deriv.back() = ((h_lower + h_upper) * deriv[grid.size() - 2]
                        - h_upper * deriv[grid.size() - 3])
                       / h_lower;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives.
 *
 * This is a hack to produce the same interpolation results as Geant4. The
 * calculation here is identical to Geant4's \c
 * G4PhysicsVector::ComputeSecDerivative1, which is based off the algorithm for
 * calculating the second derivatives of a cubic spline in Numerical Recipes,
 * modified for not-a-knot boundary conditions.
 *
 * Note that here the coefficients are divided by \f$ h_i + h_{i + 1} \f$.
 *
 * \todo While Geant4 supposedly uses not-a-knot boundary conditions, these
 * second derivatives differ from the expected values.
 */
template<class GA>
auto SplineDerivCalculator::calc_geant_derivatives(GA const& grid) const
    -> VecReal
{
    size_type num_knots = grid.size();

    // Used to store the result as well as temporary storage for the decomposed
    // factors in the tridiagonal algorithm
    VecReal result(num_knots);
    VecReal rhs(num_knots - 1);

    // Set up the initial not-a-knot boundary conditions
    real_type h_lower = grid.delta_x(0);
    real_type h_upper = grid.delta_x(1);

    // First \c c_prime value (negated) for the tridiagonal algorithm: \f$ -c'
    // = -a_2 / a_1 \f$.
    result[1] = (h_lower - h_upper) / (2 * h_upper + h_lower);

    // XXX Almost \f$ a_3 / a_1 \f$ (which would be \f$ 6 r_0 h_1 / ((h_0 + 2
    // h_1)(h_0 + h_1)) \f$ )
    rhs[1] = 6 * grid.delta_slope(1) * h_upper / ipow<2>(h_lower + h_upper);

    // Tridiagonal algorithm decomposition and forward substitution
    for (size_type i = 2; i < num_knots - 2; ++i)
    {
        // Calculate the coefficients while performing the forward sweep
        h_lower = grid.delta_x(i - 1);
        h_upper = grid.delta_x(i);

        // \f$ a_0 = h_{i - 1} / (h_{i - 1} + h_i) \f$
        real_type sig = h_lower / (h_lower + h_upper);

        // \f$ p = 1 / (a_1 - a_0 c'_{i - 1}) \f$
        real_type p = 1 / (2 + sig * result[i - 1]);

        // \f$ -c'_i = -a_2 p = h_{i} / ((h_{i - 1} + h_i) p) \f$
        result[i] = (sig - 1) * p;

        // XXX Almost \f$ u_i = (a_3 - a_0 u_{i - 1}) p \f$ (note that the RHS
        // a_3 is not multiplied by p)
        rhs[i] = 6 * grid.delta_slope(i) / (h_lower + h_upper)
                 - sig * rhs[i - 1] * p;
    }

    // Set up the final not-a-knot boundary conditions
    h_lower = grid.delta_x(num_knots - 3);
    h_upper = grid.delta_x(num_knots - 2);

    // XXX Calculate the next-to-last derivative outside of the back
    // substitution loop
    real_type sig = h_lower / (h_lower + h_upper);
    real_type p = 1 / (2 + sig * result[num_knots - 3]);
    rhs[num_knots - 2] = 6 * grid.delta_slope(num_knots - 2) * sig
                             / (h_lower + h_upper)
                         - (2 * sig - 1) * rhs[num_knots - 3] * p;
    p = 1 / ((1 + sig) + (2 * sig - 1) * result[num_knots - 3]);
    result[num_knots - 2] = rhs[num_knots - 2] * p;

    // XXX Back substitution
    for (size_type i = num_knots - 3; i >= 1; --i)
    {
        h_lower = grid.delta_x(i - 1);
        h_upper = grid.delta_x(i);
        result[i] *= result[i + 1] - rhs[i] * (h_lower + h_upper) / h_upper;
    }

    // Recover \f$ S''_0 \f$ and \f$ S''_{n - 1} \f$
    this->calc_boundaries(grid, result);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
