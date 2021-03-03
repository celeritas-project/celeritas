//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "FieldStepper.hh"
#include "FieldEquation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * This is the 4th order classical Runge-Kutta stepper
 */
class RungeKutta : public FieldStepper<RungeKutta>
{
    using ode_type = OdeArray;

  public:
    // Construct with the equation of motion
    inline CELER_FUNCTION RungeKutta(FieldEquation& equation);

    // Mandatory method - static inheritance
    CELER_FUNCTION void ode_stepper(real_type       h,
                                    const ode_type& y,
                                    const ode_type& dydx,
                                    ode_type&       yout);

  private:
    // The base class is a friend of this
    friend class FieldStepper<RungeKutta>;

  private:
    ode_type dydxm_;
    ode_type dydxt_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RungeKutta.i.hh"
