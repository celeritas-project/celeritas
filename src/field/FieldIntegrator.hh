//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldIntegrator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

#include "RungeKutta.hh"
#include "FieldStepper.hh"
#include "FieldParamsPointers.hh"
#include "field/base/OdeArray.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 *  This is a driver to control the quality of the field integration stepper
 *  and provides the integration with a given field stepper
 *
 * \note This class is based on G4MagIntegratorDriver and and G4ChordFinder
 */
class FieldIntegrator
{
    using ode_type = OdeArray;

  public:
    // Construct with shared data and the stepper
    inline CELER_FUNCTION
    FieldIntegrator(const FieldParamsPointers& shared, RungeKutta& stepper);

    // Interfaces from G4ChordFinder
    CELER_FUNCTION real_type advance_chord_limited(real_type step_trial,
                                                   ode_type& y);

    CELER_FUNCTION real_type find_next_chord(real_type       hstep,
                                             const ode_type& y,
                                             ode_type&       yend,
                                             real_type&      dyerr);

    // An adaptive stepsize control from G4MagIntegratorDriver
    CELER_FUNCTION bool accurate_advance(real_type  hstep,
                                         ode_type&  y,
                                         real_type& curveLength,
                                         real_type  hinitial);

    CELER_FUNCTION real_type one_good_step(real_type       hstep,
                                           ode_type&       y,
                                           const ode_type& dydx,
                                           real_type&      hnext);

    // advance for the small step
    CELER_FUNCTION real_type quick_advance(real_type       hstep,
                                           ode_type&       y,
                                           const ode_type& dydx,
                                           real_type&      dchord_step);

    CELER_FUNCTION real_type new_step_size(real_type hstep, real_type error);

    CELER_FUNCTION void ode_rhs(const ode_type y, ode_type& dydx)
    {
        stepper_.ode_rhs(y, dydx);
    }

    // >>> COMMON PROPERTIES

    //! XXX move to constants?
    static CELER_CONSTEXPR_FUNCTION real_type permillion() { return 1e-6; }

  private:
    // Shared constant properties
    const FieldParamsPointers& shared_;

    // Stepper for this field driver
    RungeKutta& stepper_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldIntegrator.i.hh"
