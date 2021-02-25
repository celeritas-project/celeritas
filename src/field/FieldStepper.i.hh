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
real_type FieldStepper<T>::stepper(real_type       hstep,
                                   const ode_type  y,
                                   const ode_type& dydx,
                                   ode_type&       yout)
{
    // Do two half steps
    ode_stepper(0.5 * hstep, y, dydx, ymid);
    equation_(ymid, dydxmid);
    ode_stepper(0.5 * hstep, ymid, dydxmid, yout);

    // Do a full step
    ode_stepper(hstep, y, dydx, yt);

    // Stepper error: difference between the full step and two half steps
    ode_type yerr = yout - yt;

    // Output correction with the 4th order coefficient (1/15)
    yout += yerr / 15.;

    // Evaluate tolerance and squre of the position accuracy
    real_type eps_pos = eps_rel_max() * hstep;
    real_type errpos2 = yerr.position_square();

    // Scale relative to a required tolerance
    CELER_ASSERT(errpos2 >= 0.0);
    errpos2 /= (eps_pos * eps_pos);

    // Evaluate tolerance and squre of the momentum accuracy
    real_type magvel2 = y.momentum_square();
    real_type errvel2 = yerr.momentum_square();

    CELER_ASSERT(magvel2 > 0.0);
    errvel2 /= (magvel2 * eps_rel_max() * eps_rel_max());

    // Return the square of the maximum truncation error
    return std::fmax(errpos2, errvel2);
}

template<typename T>
real_type FieldStepper<T>::distance_chord(const ode_type y, const ode_type yout)
{
    return ymid.distance_closest(y, yout);
}

template<typename T>
void FieldStepper<T>::ode_rhs(ode_type y, ode_type& yout)
{
    equation_(y, yout);
}
//---------------------------------------------------------------------------//
} // namespace celeritas
