//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceIntegrator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Integrate the field ODEs using Dormand-Prince RK5(4)7M.
 *
 * The algorithm, RK5(4)7M and the coefficients have been adapted from
 * J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae"
 * Journal Computational and Applied Mathematics, volume 6, no 1 (1980) and
 * the coefficients to locate the mid point are taken from
 * \citet{shampine-rungekutta-1986, https://doi.org/10.2307/2008219}.
 *
 * For a given ordinary differential equation, \f$\difd{y}{x} = f(x, y)\f$,
 * the
 * fifth order solution, \f$ y_{n+1} \f$, an embedded fourth order solution,
 * \f$ y^{*}_{n+1} \f$, and the error estimate as difference between them are
 * as follows,
 * \f[
 * \begin{aligned}
     y_{n+1}     &= y_n + h \sum_{n=1}^{6} b_i  k_i + O(h^6) \\
     y^{*}_{n+1} &= y_n + h \sum_{n=1}^{7} b*_i k_i + O(h^5) \\
     y_{error}   &= y_{n+1} - y^{*}_{n+1} = \sum_{n=1}^{7} (b^{*}_i - b_i) k_i
 * \end{aligned}
 * \f]
 * where \f$h\f$ is the step to advance and \f$k_i\f$ is the right hand side of
 * the function at \f$x_n + h c_i\f$,
 * and the coefficients (The Butcher table) for Dormand-Prince RK5(4)7M are
 * \verbatim
   c_i | a_ij
   0   |
   1/5 | 1/5
   3/10| 3/40        9/40
   4/5 | 44/45      -56/15      32/9
   8/9 | 19372/6561 -25360/2187 64448/6561 -212/729
   1   | 9017/3168  -355/33     46732/5247  49/176  -5103/18656
   1   | 35/384      0          500/1113    125/192 -2187/6784    11/84
  ----------------------------------------------------------------------------
   b*_i| 35/384      0          500/1113    125/192 -2187/6784    11/84    0
   b_i | 5179/57600  0          7571/16695  393/640 -92097/339200 187/2100 1/40
  \endverbatim
 *
 * The result at the mid point is computed
 * \f[
     y_{n+1/2}  = y_n + (h/2) \sum_{n=1}^{7} c^{*}_i k_i
   \f]
 * with the coefficients \f$c^{*}\f$ taken from Shampine.
 *
 * \note The analogous class in Geant4 is \c G4DormandPrince745 , which
 inherits from \c G4MagIntegratorStepper.
 */
template<class EquationT>
class DormandPrinceIntegrator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = FieldIntegration;
    //!@}

  public:
    //! Construct with the equation of motion
    explicit CELER_FUNCTION DormandPrinceIntegrator(EquationT&& eq)
        : calc_rhs_(::celeritas::forward<EquationT>(eq))
    {
    }

    // Adaptive step size control
    CELER_FUNCTION result_type operator()(real_type step,
                                          OdeState const& beg_state) const;

  private:
    // Functor to calculate the force applied to a particle
    EquationT calc_rhs_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EquationT>
CELER_FUNCTION
DormandPrinceIntegrator(EquationT&&) -> DormandPrinceIntegrator<EquationT>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Numerically integrate using the DormandPrince RK5(4)7M method.
 */
template<class E>
CELER_FUNCTION auto DormandPrinceIntegrator<E>::operator()(
    real_type step, OdeState const& beg_state) const -> result_type
{
    using celeritas::axpy;
    using R = real_type;

    // Coefficients for Dormand-Prince RK5(4)7M
    constexpr R a11 = 0.2;

    constexpr R a21 = 0.075;
    constexpr R a22 = 0.225;

    constexpr R a31 = 44 / R(45);
    constexpr R a32 = -56 / R(15);
    constexpr R a33 = 32 / R(9);

    constexpr R a41 = 19372 / R(6561);
    constexpr R a42 = -25360 / R(2187);
    constexpr R a43 = 64448 / R(6561);
    constexpr R a44 = -212 / R(729);

    constexpr R a51 = 9017 / R(3168);
    constexpr R a52 = -355 / R(33);
    constexpr R a53 = 46732 / R(5247);
    constexpr R a54 = 49 / R(176);
    constexpr R a55 = -5103 / R(18656);

    constexpr R a61 = 35 / R(384);
    constexpr R a63 = 500 / R(1113);
    constexpr R a64 = 125 / R(192);
    constexpr R a65 = -2187 / R(6784);
    constexpr R a66 = 11 / R(84);

    constexpr R d71 = a61 - 5179 / R(57600);
    constexpr R d73 = a63 - 7571 / R(16695);
    constexpr R d74 = a64 - 393 / R(640);
    constexpr R d75 = a65 + 92097 / R(339200);
    constexpr R d76 = a66 - 187 / R(2100);
    constexpr R d77 = -1 / R(40);

    // Coefficients for the mid point calculation by Shampine
    constexpr R c71 = R(6025192743.) / R(30085553152.);
    constexpr R c73 = R(51252292925.) / R(65400821598.);
    constexpr R c74 = R(-2691868925.) / R(45128329728.);
    constexpr R c75 = R(187940372067.) / R(1594534317056.);
    constexpr R c76 = R(-1776094331.) / R(19743644256.);
    constexpr R c77 = R(11237099.) / R(235043384.);

    result_type result;

    // First step
    OdeState k1 = calc_rhs_(beg_state);
    OdeState state = beg_state;
    axpy(a11 * step, k1, &state);

    // Second step
    OdeState k2 = calc_rhs_(state);
    state = beg_state;
    axpy(a21 * step, k1, &state);
    axpy(a22 * step, k2, &state);

    // Third step
    OdeState k3 = calc_rhs_(state);
    state = beg_state;
    axpy(a31 * step, k1, &state);
    axpy(a32 * step, k2, &state);
    axpy(a33 * step, k3, &state);

    // Fourth step
    OdeState k4 = calc_rhs_(state);
    state = beg_state;
    axpy(a41 * step, k1, &state);
    axpy(a42 * step, k2, &state);
    axpy(a43 * step, k3, &state);
    axpy(a44 * step, k4, &state);

    // Fifth step
    OdeState k5 = calc_rhs_(state);
    state = beg_state;
    axpy(a51 * step, k1, &state);
    axpy(a52 * step, k2, &state);
    axpy(a53 * step, k3, &state);
    axpy(a54 * step, k4, &state);
    axpy(a55 * step, k5, &state);

    // Sixth step
    OdeState k6 = calc_rhs_(state);
    result.end_state = beg_state;
    axpy(a61 * step, k1, &result.end_state);
    axpy(a63 * step, k3, &result.end_state);
    axpy(a64 * step, k4, &result.end_state);
    axpy(a65 * step, k5, &result.end_state);
    axpy(a66 * step, k6, &result.end_state);

    // Seventh step: the final step
    OdeState k7 = calc_rhs_(result.end_state);

    // The error estimate
    result.err_state = {{0, 0, 0}, {0, 0, 0}};
    axpy(d71 * step, k1, &result.err_state);
    axpy(d73 * step, k3, &result.err_state);
    axpy(d74 * step, k4, &result.err_state);
    axpy(d75 * step, k5, &result.err_state);
    axpy(d76 * step, k6, &result.err_state);
    axpy(d77 * step, k7, &result.err_state);

    // The mid point
    real_type half_step = step / real_type(2);
    result.mid_state = beg_state;
    axpy(c71 * half_step, k1, &result.mid_state);
    axpy(c73 * half_step, k3, &result.mid_state);
    axpy(c74 * half_step, k4, &result.mid_state);
    axpy(c75 * half_step, k5, &result.mid_state);
    axpy(c76 * half_step, k6, &result.mid_state);
    axpy(c77 * half_step, k7, &result.mid_state);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
