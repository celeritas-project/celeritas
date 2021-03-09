//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldStepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "FieldEquation.hh"

#include "field/base/OdeArray.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The base class of Runga-Kutta family steppers
 *
 * \tparam T A derived class.
 */
template<typename T>
class FieldStepper
{
    using ode_type = OdeArray<real_type, 6>;

  public:
    // Construct with the equation of motion
    inline CELER_FUNCTION FieldStepper(FieldEquation& equation);

    // Commom methods

    // Adaptive step size control
    CELER_FUNCTION ode_type operator()(real_type       h,
                                       const ode_type& y,
                                       const ode_type& dydx);

    // Stepper truncation error
    CELER_FUNCTION real_type error(real_type h, const ode_type& y);

    // Closerest distance between the chord and the mid-point
    CELER_FUNCTION real_type distance_chord(const ode_type& y,
                                            const ode_type& yout) const
    {
        return ymid_.distance_closest(y, yout);
    }

    // The right hand side of the field equation
    CELER_FUNCTION ode_type ode_rhs(const ode_type& y) const
    {
        return equation_(y);
    }

    // Static interfaces (Mandatory methods)
    CELER_FUNCTION ode_type stepper(real_type       h,
                                    const ode_type& y,
                                    const ode_type& dydx)
    {
        return static_cast<T*>(this)->stepper(h, y, dydx);
    }

    // >>> COMMON PROPERTIES

    //! Maximum relative error scale
    static CELER_CONSTEXPR_FUNCTION real_type eps_rel_max() { return 1e-3; }

  protected:
    // Equation of the motion
    FieldEquation& equation_;

    // States at the middle point and at the end of one full step
    ode_type ymid_;
    ode_type dydxmid_;
    ode_type yt_;

    // Truncation error
    ode_type yerr_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldStepper.i.hh"
