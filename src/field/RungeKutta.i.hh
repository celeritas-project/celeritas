//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
RungeKutta::RungeKutta(FieldEquation& equation)
    : FieldStepper<RungeKutta>(equation)
{
}

//---------------------------------------------------------------------------//
/*!
 * The classical RungeKutta stepper (the 4th order)
 */
void RungeKutta::ode_stepper(real_type       h,
                             const ode_type  y,
                             const ode_type& dydx,
                             ode_type&       yout)
{
    // 1st step k1 = (h/2)*dydx
    yt = y + 0.5 * h * dydx;

    // 2nd step k2 = (h/2)*dydxt
    equation_(yt, dydxt);
    yt = y + 0.5 * h * dydxt;

    // 3rd step k3 = h*dydxm
    equation_(yt, dydxm);
    yt = y + h * dydxm;

    // now dydxm = (k2+k3)/h
    dydxm += dydxt;

    // 4th Step k4 = h*dydxt and the final RK4 output: k1/6+k4/6+(k2+k3)/3
    equation_(yt, dydxt);
    yout = y + h * (dydx + dydxt + 2.0 * dydxm) / 6.0;
}
//---------------------------------------------------------------------------//
} // namespace celeritas
