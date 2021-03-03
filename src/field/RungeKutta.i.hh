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
                             const ode_type& y,
                             const ode_type& dydx,
                             ode_type&       yout)
{
    // 1st step k1 = (h/2)*dydx
    yt_ = y + 0.5 * h * dydx;

    // 2nd step k2 = (h/2)*dydxt
    dydxt_ = equation_(yt_);
    yt_    = y + 0.5 * h * dydxt_;

    // 3rd step k3 = h*dydxm
    dydxm_ = equation_(yt_);
    yt_    = y + h * dydxm_;

    // now dydxm = (k2+k3)/h
    dydxm_ += dydxt_;

    // 4th Step k4 = h*dydxt and the final RK4 output: k1/6+k4/6+(k2+k3)/3
    dydxt_ = equation_(yt_);
    yout   = y + h * (dydx + dydxt_ + 2.0 * dydxm_) / 6.0;
}
//---------------------------------------------------------------------------//
} // namespace celeritas
