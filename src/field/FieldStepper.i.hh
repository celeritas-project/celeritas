//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldStepper.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the equation of motion.
 */
template<typename T>
CELER_FUNCTION FieldStepper<T>::FieldStepper(FieldEquation& equation)
    : equation_(equation)
{
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step size control
 */
template<typename T>
OdeArray<real_type, 6> FieldStepper<T>::
                       operator()(real_type hstep, const ode_type& y, const ode_type& dydx)
{
    // Do two half steps
    ymid_         = stepper(0.5 * hstep, y, dydx);
    dydxmid_      = equation_(ymid_);
    ode_type yout = stepper(0.5 * hstep, ymid_, dydxmid_);

    // Do a full step
    yt_ = stepper(hstep, y, dydx);

    // Stepper error: difference between the full step and two half steps
    yerr_ = yout - yt_;

    // Output correction with the 4th order coefficient (1/15)
    yout += yerr_ / 15.;

    return yout;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the stepper truncation error: max(pos_error^2, scale*mom_error^2)
 */
template<typename T>
real_type FieldStepper<T>::error(real_type hstep, const ode_type& y)
{
    // Evaluate tolerance and squre of the position accuracy
    real_type eps_pos = eps_rel_max() * hstep;
    real_type errpos2 = yerr_.position_square();

    // Scale relative to a required tolerance
    CELER_ASSERT(errpos2 >= 0.0);
    errpos2 /= (eps_pos * eps_pos);

    // Evaluate tolerance and squre of the momentum accuracy
    real_type magvel2 = y.momentum_square();
    real_type errvel2 = yerr_.momentum_square();

    CELER_ASSERT(magvel2 > 0.0);
    errvel2 /= (magvel2 * eps_rel_max() * eps_rel_max());

    // Return the square of the maximum truncation error
    return std::fmax(errpos2, errvel2);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
