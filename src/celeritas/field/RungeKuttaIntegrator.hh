//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RungeKuttaIntegrator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "Types.hh"

#include "detail/FieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Integrate the field ODEs using the 4th order classical Runge-Kutta method.
 *
 * This method estimates the updated state from an initial state and evaluates
 * the truncation error, with fourth-order accuracy based on description in
 * Numerical Recipes in C, The Art of Scientific Computing, Sec. 16.2,
 * Adaptive Stepsize Control for Runge-Kutta.
 *
 * For a magnetic field equation \em f along a charged particle trajectory
 * with state \em y, which includes position and momentum but may also include
 * time and spin. For N-variables (\em i = 1, ... N), the right hand side of
 * the equation
 * \f[
 *  \difd{y_{i}}{s} = f_i (s, y_{i})
 * \f]
 * and the fouth order Runge-Kutta solution for a given step size, \em h is
 * \f[
 *  y_{n+1} - y_{n} = h f(x_n, y_n) = \frac{h}{6} (k_1 + 2 k_2 + 2 k_3 + k_4)
 * \f]
 * which is the average slope at four different points.
 * The truncation error is the difference of the final states of one full step
 * (\em y1) and two half steps (\em y2)
 * \f[
 *  \Delta = y_2 - y_1, y(x+h) = y_2 + \frac{\Delta}{15} + \mathrm{O}(h^{6})
 * \f]
 */
template<class EquationT>
class RungeKuttaIntegrator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = FieldIntegration;
    //!@}

  public:
    //! Construct with the equation of motion
    explicit CELER_FUNCTION RungeKuttaIntegrator(EquationT&& eq)
        : calc_rhs_(::celeritas::forward<EquationT>(eq))
    {
    }

    // Advance the ODE state according to the field equations
    CELER_FUNCTION result_type operator()(real_type step,
                                          OdeState const& beg_state) const;

  private:
    // Return the final state by the 4th order Runge-Kutta method
    CELER_FUNCTION auto do_step(real_type step,
                                OdeState const& beg_state,
                                OdeState const& end_slope) const -> OdeState;

  private:
    // Equation of the motion
    EquationT calc_rhs_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EquationT>
CELER_FUNCTION RungeKuttaIntegrator(EquationT&&)
    -> RungeKuttaIntegrator<EquationT>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Numerically integrate and return the updated state with estimated error.
 */
template<class E>
CELER_FUNCTION auto
RungeKuttaIntegrator<E>::operator()(real_type step,
                                    OdeState const& beg_state) const
    -> result_type
{
    using celeritas::axpy;
    real_type half_step = step / real_type(2);
    constexpr real_type fourth_order_correction = 1 / real_type(15);

    result_type result;
    OdeState beg_slope = calc_rhs_(beg_state);

    // Do two half steps
    result.mid_state = this->do_step(half_step, beg_state, beg_slope);
    result.end_state = this->do_step(
        half_step, result.mid_state, calc_rhs_(result.mid_state));

    // Do a full step
    OdeState yt = this->do_step(step, beg_state, beg_slope);

    // Integrator error: difference between the full step and two half steps
    result.err_state = result.end_state;
    axpy(real_type(-1), yt, &result.err_state);

    // Output correction with the 4th order coefficient (1/15)
    axpy(fourth_order_correction, result.err_state, &result.end_state);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * The classical RungeKuttaIntegrator integrator (the 4th order).
 */
template<class E>
CELER_FUNCTION auto
RungeKuttaIntegrator<E>::do_step(real_type step,
                                 OdeState const& beg_state,
                                 OdeState const& beg_slope) const -> OdeState
{
    using celeritas::axpy;
    real_type half_step = step / real_type(2);
    constexpr real_type sixth = 1 / real_type(6);

    // 1st step k1 = (step/2)*beg_slope
    OdeState mid_est = beg_state;
    axpy(half_step, beg_slope, &mid_est);
    OdeState mid_est_slope = calc_rhs_(mid_est);

    // 2nd step k2 = (step/2)*mid_est_slope
    OdeState mid_state = beg_state;
    axpy(half_step, mid_est_slope, &mid_state);
    OdeState mid_slope = calc_rhs_(mid_state);

    // 3rd step k3 = step*mid_slope
    OdeState end_est = beg_state;
    axpy(step, mid_slope, &end_est);
    OdeState end_slope = calc_rhs_(end_est);

    // Average slope at all 4 points
    axpy(real_type(1), beg_slope, &end_slope);
    axpy(real_type(2), mid_slope, &end_slope);
    axpy(real_type(2), mid_est_slope, &end_slope);

    // 4th Step k4 = h*dydxt and the final RK4 output: k1/6+k4/6+(k2+k3)/3
    OdeState end_state = beg_state;
    axpy(sixth * step, end_slope, &end_state);

    return end_state;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
