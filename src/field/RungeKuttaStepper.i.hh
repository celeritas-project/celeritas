//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKuttaStepper.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a field equation.
 */
template<class FieldEquation_T>
CELER_FUNCTION
RungeKuttaStepper<FieldEquation_T>::RungeKuttaStepper(FieldEquation_T& eq)
    : equation_(eq)
{
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step size control
 */
template<class T>
CELER_FUNCTION 
auto RungeKuttaStepper<T>::operator()(real_type step, 
                                      const OdeState& beg_state, 
                                      const OdeState& beg_slope) -> OdeState
{
    using celeritas::axpy;
    real_type           half_step               = step / real_type(2);
    constexpr real_type fourth_order_correction = 1 / real_type(15);

    // Do two half steps
    mid_state_         = stepper(half_step, beg_state, beg_slope);
    OdeState end_state = stepper(half_step, mid_state_, equation_(mid_state_));

    // Do a full step
    OdeState yt = stepper(step, beg_state, beg_slope);

    // Stepper error: difference between the full step and two half steps
    state_err_ = end_state;
    axpy(real_type(-1.0), yt, &state_err_);

    // Output correction with the 4th order coefficient (1/15)
    axpy(fourth_order_correction, state_err_, &end_state);

    return end_state;
}

//---------------------------------------------------------------------------//
/*!
 * The classical RungeKuttaStepper stepper (the 4th order)
 */
template<class T>
CELER_FUNCTION auto RungeKuttaStepper<T>::stepper(real_type       step,
                                                  const OdeState& beg_state,
                                                  const OdeState& beg_slope)
    -> OdeState
{
    using celeritas::axpy;
    real_type           half_step = step / real_type(2);
    constexpr real_type sixth     = 1 / real_type(6);

    // 1st step k1 = (step/2)*beg_slope
    OdeState mid_est = beg_state;
    axpy(half_step, beg_slope, &mid_est);
    OdeState mid_est_slope = equation_(mid_est);

    // 2nd step k2 = (step/2)*mid_est_slope
    OdeState mid_state = beg_state;
    axpy(half_step, mid_est_slope, &mid_state);
    OdeState mid_slope = equation_(mid_state);

    // 3rd step k3 = step*mid_slope
    OdeState end_est = beg_state;
    axpy(step, mid_slope, &end_est);
    OdeState end_slope = equation_(end_est);

    // Average slope at all 4 points
    axpy(real_type(1.0), beg_slope, &end_slope);
    axpy(real_type(2.0), mid_slope, &end_slope);
    axpy(real_type(2.0), mid_est_slope, &end_slope);

    // 4th Step k4 = h*dydxt and the final RK4 output: k1/6+k4/6+(k2+k3)/3
    OdeState end_state = beg_state;
    axpy(sixth * step, end_slope, &end_state);

    return end_state;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the stepper truncation error: max(pos_error^2, scale*mom_error^2)
 */
template<class T>
CELER_FUNCTION real_type RungeKuttaStepper<T>::error(real_type       step,
                                                     const OdeState& beg_state)
{
    // Evaluate tolerance and squre of the position and momentum accuracy
    real_type eps_pos = eps_rel_max() * step;

    real_type magvel2{0.0};
    real_type errpos2{0.0};
    real_type errvel2{0.0};

    for (auto i : range(3))
    {
        magvel2 += beg_state[i + 3] * beg_state[i + 3];
        errpos2 += state_err_[i] * state_err_[i];
        errvel2 += state_err_[i + 3] * state_err_[i + 3];
    }

    // Scale relative to a required tolerance
    CELER_ASSERT(errpos2 > 0.0);
    CELER_ASSERT(magvel2 > 0.0);

    errpos2 /= (eps_pos * eps_pos);
    errvel2 /= (magvel2 * eps_rel_max() * eps_rel_max());

    // Return the square of the maximum truncation error
    return std::fmax(errpos2, errvel2);
}

//---------------------------------------------------------------------------//
/*!
 * Closerest distance between the chord and the mid-point
 */
template<class T>
CELER_FUNCTION real_type RungeKuttaStepper<T>::distance_chord(
    const OdeState& beg_state, const OdeState& end_state) const
{
    real_type beg_dist2{0};
    real_type end_dist2{0};

    for (auto i : range(3))
    {
        beg_dist2 += beg_state[i] * beg_state[i];
        end_dist2 += end_state[i] * end_state[i];
    }

    return std::sqrt(beg_dist2 * end_dist2 / (beg_dist2 + end_dist2));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
