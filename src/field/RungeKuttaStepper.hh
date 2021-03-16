//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKuttaStepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Macros.hh"
//#include "FieldEquation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The 4th order classical Runge-Kutta stepper
 *
 * This method estimates the updated state from an initial state and slope
 * estimate, with fourth-order accuracy.
 *
 * For a magnetic field equation \em f with state \em x, which includes
 * position and momentum but may also include time and spin:
 * \f[
 *  XXX TODO: document steps here in latex math
 * \f]
 */
template<class FieldEquation_T>
class RungeKuttaStepper
{
    //@{
    //! Type aliases
    using OdeState = typename FieldEquation_T::OdeState;
    //@}

  public:
    //! Construct with the equation of motion
    inline CELER_FUNCTION RungeKuttaStepper(FieldEquation_T& equation);

    //! Adaptive step size control
    CELER_FUNCTION auto operator()(real_type       step,
                                   const OdeState& beg_state,
                                   const OdeState& beg_slope) -> OdeState;

    //! Stepper truncation error
    CELER_FUNCTION real_type error(real_type step, const OdeState& beg_state);

    // Closerest distance between the chord and the mid-point
    CELER_FUNCTION real_type distance_chord(const OdeState& beg_state,
                                            const OdeState& end_state) const;

    //! The right hand side of the field equation
    CELER_FUNCTION OdeState ode_rhs(const OdeState& beg_state) const
    {
        return equation_(beg_state);
    }

  private:
    //! return the final state by the 4th order Runge-Kutta method
    CELER_FUNCTION auto stepper(real_type       step,
                                const OdeState& beg_state,
                                const OdeState& end_slope) -> OdeState;

    //! Maximum relative error scale
    static CELER_CONSTEXPR_FUNCTION real_type eps_rel_max() { return 1e-3; }

  private:
    //! Equation of the motion
    FieldEquation_T& equation_;

    //! State at the middle point
    OdeState mid_state_;

    //! Stepper truncation error at the end of the full step
    OdeState state_err_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RungeKuttaStepper.i.hh"
