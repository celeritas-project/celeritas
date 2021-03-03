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

    // For a given trial step (hstep), advance by a sub_step within
    // a required tolerence error and update current states (y)
    CELER_FUNCTION real_type operator()(real_type hstep, ode_type& y);

    // Find the next acceptable chord of which sagitta is smaller than a
    // given miss-distance (delta_chord) and evaluate the assocated error
    CELER_FUNCTION real_type find_next_chord(real_type       hstep,
                                             const ode_type& y,
                                             ode_type&       yend,
                                             real_type&      dyerr);

    // An adaptive stepsize control from G4MagIntegratorDriver
    CELER_FUNCTION bool accurate_advance(real_type  hstep,
                                         ode_type&  y,
                                         real_type& curveLength,
                                         real_type  hinitial);

    // Avance based on the miss distance and an associated stepper error
    CELER_FUNCTION real_type quick_advance(real_type       hstep,
                                           ode_type&       y,
                                           const ode_type& dydx,
                                           real_type&      dchord_step);

  private:
    // Advance within the truncated error and estimate a good next step size
    CELER_FUNCTION real_type one_good_step(real_type       hstep,
                                           ode_type&       y,
                                           const ode_type& dydx,
                                           real_type&      hnext);

    // Find the next chord and return a step length taken and updates the state
    CELER_FUNCTION real_type new_step_size(real_type hstep,
                                           real_type error) const;

    // >>> COMMON PROPERTIES

    static CELER_CONSTEXPR_FUNCTION real_type rel_tolerance() { return 1e-6; }

    CELER_FUNCTION bool check_sagitta(real_type       hstep,
                                      const ode_type& y,
                                      const ode_type& dydx,
                                      ode_type&       yend,
                                      real_type&      dyerr,
                                      real_type&      dchord);

    CELER_FUNCTION bool move_step(real_type  hstep,
                                  real_type  h_threshold,
                                  real_type  end_curve_length,
                                  ode_type&  y,
                                  real_type& hnext,
                                  real_type& curveLength);

  private:
    // Shared constant properties
    const FieldParamsPointers& shared_;

    // Stepper for this field driver
    RungeKutta& stepper_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldIntegrator.i.hh"
