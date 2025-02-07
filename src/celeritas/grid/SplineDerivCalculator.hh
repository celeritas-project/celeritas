//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineDerivCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/grid/UniformGrid.hh"

#include "detail/GridAccessor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives of a cubic spline.
 *
 * See section 3.3: Cubic Spline Interpolation in \cite{press-nr-1992} for a
 * review of interpolating cubic splines and an algorithm for calculating the
 * second derivatives.
 *
 * Determining the polynomial coefficients \$ a_0, a_1, a_2 \f$ and \$f a_3 \f$
 * of a cubic spline \f$ S(x) \f$ (see: \sa SplineInterpolator) requires
 * solving a tridiagonal, linear system of equations for the second
 * derivatives. For \f$ n \f$ points \f$ (x_i, y_i) \f$ and \f$ n \f$ unknowns
 * \f$ S''_i \f$ there are \f$ n - 2 \f$ equations of the form
 * \f[
   h_{i - 1} S''_{i - 1} + 2 (h_{i - 1} + h_i) S''i + h_i S''_{i + 1} = 6 r_i,
 * \f]
 * where \f$ r_i = frac{\Delta y_i}{h_i} - frac{\Delta y_{i - 1}}{h_{i - 1}}
 * \f$ and \f$ h_i = \Delta x_i = x_{i + 1} - x_i \f$.
 *
 * Specifying the boundary conditions gives the remaining two equations.
 * Natural boundary conditions set \f$ S''_0 = S''_{n - 1} = 0 \f$, which leads
 * to the following initial and final equations:
 * \f{align}{
   2 (h_0 + h_1) S''_1 + h_1 S''_2 &= 6 r_1 //
   h_{n - 3} S''_{n - 3} + 2 (h_{n - 3} + h_{n - 2}) S''_{n - 2}
   &= 6 r_{n - 2}.
 * \f}
 *
 * The points \f$ x_0, x_1, \dots , x_{n - 1} \f$ where the spline changes from
 * one cubic to the next are called knots. "Not-a-knot" boundary conditions
 * require the third derivative \f$ S'''_i \f$ to be continous across the first
 * and final interior knots, \f$ x_1 \f$ and \f$ x_{n - 2} \f$ (the name refers
 * to the polynomials on the interval \f$ (x_0, x_1) \f$  and \f$ (x_1, x_2)
 * \f$ being the same cubic, so \f$ x_1 \f$ is "not a knot"). This constraint
 * gives the final two equations:
 * \f{align}{
   \frac{(h_0 + 2 h_1)(h_0 + h_1)}{h_1} S''_1 + \frac{h_1^2 - h_0^2}{h_1}
   S''_2 &= 6 r_1 //
   \frac{h_{n - 3}^2 - h_{n - 2}^2}{h_{n - 3}} S''_{n - 3} + \frac{(h_{n - 3}
   + h_{n - 2})(2 h_{n - 3} + h_{n - 2})}{h_{n - 3}} S''_{n - 2} &= 6 r_{n - 2}
 * \f}
 * Once the system of equations has been solved for the second derivatives, the
 * derivatives \f$ S''_0 \f$ and \f$ S''_{n - 1} \f$ can be recovered:
 * \f{align}{
   S''_0 &= \frac{(h_0 + h_1) S''_1 - h_0 S''_2}{h_1} \\
   S''_{n - 1} &= \frac{(h_{n - 3} + h_{n - 2}) S''_{n - 2} - h_{n - 2}
   S''_{n - 3}}{h_{n - 3}}
 * \f}
 */
class SplineDerivCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstReal = detail::SpanGridAccessor::SpanConstReal;
    using Values = detail::XsGridAccessor::Values;
    using VecReal = std::vector<real_type>;
    //!@}

    //! Cubic spline interpolation boundary conditions
    enum class BoundaryCondition
    {
        natural = 0,
        not_a_knot,
        geant,  //!< Geant4's "not-a-knot"
        size_
    };

  public:
    // Construct with boundary conditions
    explicit SplineDerivCalculator(BoundaryCondition);

    // Calculate the second derivatives
    VecReal operator()(SpanConstReal, SpanConstReal) const;
    VecReal operator()(XsGridData const&, Values const&) const;

  private:
    //// TYPES ////

    using Real3 = Array<real_type, 3>;

    //// DATA ////

    BoundaryCondition bc_;

    //// HELPER FUNCTIONS ////

    template<class GridAccessor>
    VecReal operator()(GridAccessor&&) const;

    template<class GridAccessor>
    void calc_initial_coeffs(GridAccessor const&, Real3&, real_type&) const;
    template<class GridAccessor>
    void calc_final_coeffs(GridAccessor const&, Real3&, real_type&) const;
    template<class GridAccessor>
    void calc_boundaries(GridAccessor const&, VecReal&) const;
    template<class GridAccessor>
    VecReal calc_geant_derivatives(GridAccessor const&) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
