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
 * Brief class description.
 *
 * This is the base class of field steppers
 */
template <typename T>
class FieldStepper
{
  using ode_type = OdeArray;
  public:
    // Construct with the equation of motion
    inline CELER_FUNCTION FieldStepper(FieldEquation& equation);

    // Commom methods
    CELER_FUNCTION real_type stepper(real_type        h,
                                     const ode_type   y,
				     const ode_type&  dydx,
				     ode_type&        yout);

    CELER_FUNCTION real_type distance_chord(const ode_type y, 
                                            const ode_type yout);

    CELER_FUNCTION void ode_rhs(ode_type y, ode_type& yout);

    // Static interfaces (Mandatory methods)
    CELER_FUNCTION void ode_stepper(real_type        h,
                                    const ode_type   y,
				    const ode_type&  dydx,
				    ode_type&        yout)
    {
        static_cast<T *>(this)->ode_stepper(h, y, dydx, yout);
    }

  // >>> COMMON PROPERTIES

  //! Maximum relative error scale
  static CELER_CONSTEXPR_FUNCTION real_type eps_rel_max()
  {
    return 1e-3; 
  }

  protected:

    // Equation of the motion
    FieldEquation& equation_;

    // States at the middle point
    ode_type ymid;
    ode_type dydxmid;

    // State by one full step and used as a temporary in ode_stepper
    ode_type yt;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldStepper.i.hh"
