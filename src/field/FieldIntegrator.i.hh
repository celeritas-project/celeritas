//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldIntegrator.i.hh
//---------------------------------------------------------------------------//

#include "base/NumericLimits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and the stepper.
 */
CELER_FUNCTION
FieldIntegrator::FieldIntegrator(const FieldParamsPointers& shared,
                                 RungeKutta&                stepper)
    : shared_(shared), stepper_(stepper)
{
    CELER_ENSURE(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step control based on G4MagIntegratorDriver:
 * For a given trial step (hstep), advance by a sub_step within a required
 * tolerence (error) and update current states (y)
 */
CELER_FUNCTION
real_type FieldIntegrator::operator()(real_type hstep, ode_type& y)
{
    ode_type  yend         = y;
    real_type curve_length = 0;

    real_type dyerr;
    real_type h = this->find_next_chord(hstep, y, yend, dyerr);

    // evaluate a relative error
    real_type ratio = dyerr / (shared_.epsilon_step * h);

    if (ratio < 1.0)
    {
        // Accept this accuracy and update the state
        y = yend;
    }
    else
    {
        // Compute the hnext for accuracy_advance
        real_type hnext = this->new_step_size(h, ratio);

        // Advance more accurately to the "end of chord"
        bool is_good = this->accurate_advance(h, y, curve_length, hnext);
        if (!is_good)
        {
            h = curve_length;
        }
    }
    return h;
}

//---------------------------------------------------------------------------//
/*!
 * Find the next acceptable chord of which sagitta is smaller than a given
 * miss-distance (delta_chord) and evaluate the assocated error
 */
CELER_FUNCTION
real_type FieldIntegrator::find_next_chord(real_type       hstep,
                                           const ode_type& y,
                                           ode_type&       yend,
                                           real_type&      dyerr)
{
    // Try with the proposed step
    real_type h = hstep;

    ode_type dydx = stepper_.ode_rhs(y);

    real_type dchord = numeric_limits<real_type>::max();

    bool converged = this->check_sagitta(h, y, dydx, yend, dyerr, dchord);
    unsigned int remaining_steps = shared_.max_nsteps;

    while (!converged && (--remaining_steps > 0))
    {
        h *= std::fmax(std::sqrt(shared_.delta_chord / dchord), 0.5);
        converged = this->check_sagitta(h, y, dydx, yend, dyerr, dchord);
    }

    // XXX TODO: assert if not converged and handle rare cases
    CELER_ASSERT(converged);

    return h;
}

//---------------------------------------------------------------------------//
/*!
 * Advance within the truncated error and estimate a good step size for
 * the next step
 */
CELER_FUNCTION real_type FieldIntegrator::one_good_step(real_type       h,
                                                        ode_type&       y,
                                                        const ode_type& dydx,
                                                        real_type&      hnext)
{
    ode_type  yout;
    real_type errmax2 = stepper_.stepper(h, y, dydx, yout);

    unsigned int remaining_steps = shared_.max_nsteps;
    while (!(errmax2 <= 1.0) && --remaining_steps > 0)
    {
        // Step failed; compute the size of retrial step.
        real_type htemp = shared_.safety * h
                          * std::pow(errmax2, 0.5 * shared_.pshrink);

        // Truncation error too large, reduce stepsize with a low bound
        h = std::fmax(htemp, shared_.max_stepping_decrease * h);

        errmax2 = stepper_.stepper(h, y, dydx, yout);
    }
    // XXX TODO: loop check and handle rare cases if happen
    CELER_ASSERT(errmax2 <= 1.0);

    // Compute size of the next step
    if (errmax2 > shared_.errcon * shared_.errcon)
    {
        hnext = shared_.safety * h * std::pow(errmax2, 0.5 * shared_.pgrow);
    }
    else
    {
        hnext = shared_.max_stepping_increase * h;
    }

    // Update position and momentum and return step taken by this trial
    y = yout;

    return h;
}

//---------------------------------------------------------------------------//
/*!
 * Advance based on the miss distance and an associated stepper error
 */
CELER_FUNCTION
real_type FieldIntegrator::quick_advance(real_type       h,
                                         ode_type&       y,
                                         const ode_type& dydx,
                                         real_type&      dchord_step)
{
    ode_type yout;

    // Do an integration step
    real_type dyerr = stepper_.stepper(h, y, dydx, yout);

    // Estimate the curve-chord distance
    dchord_step = stepper_.distance_chord(y, yout);

    // Update new position and momentum and return the error of this step
    y = yout;

    return dyerr;
}

//---------------------------------------------------------------------------//
/*!
 * Accurate_advance for an adaptive step control
 */
CELER_FUNCTION bool FieldIntegrator::accurate_advance(real_type  hstep,
                                                      ode_type&  y,
                                                      real_type& curve_length,
                                                      real_type  hinitial)
{
    // Check validity and initialization

    CELER_ASSERT(hstep > 0);
    bool succeeded = true;

    real_type end_curve_length = curve_length + hstep;

    // set an initial proposed step and evaluate the minimum threshold
    real_type h = ((hinitial > rel_tolerance() * hstep) && (hinitial < hstep))
                      ? hinitial
                      : hstep;

    real_type h_threshold = shared_.epsilon_step * hstep;

    // Perform the integration
    real_type hnext = numeric_limits<real_type>::max();
    ;

    bool condition = this->move_step(
        h, h_threshold, end_curve_length, y, hnext, curve_length);

    unsigned int remaining_steps = shared_.max_nsteps;
    while (!condition && --remaining_steps > 0)
    {
        h = std::fmax(hnext, shared_.minimum_step);
        if (curve_length + h > end_curve_length)
        {
            h = end_curve_length - curve_length;
        }
        condition = this->move_step(
            h, h_threshold, end_curve_length, y, hnext, curve_length);
    }
    // XXX TODO: loop check and handle rare cases if happen
    CELER_ASSERT(condition);

    succeeded = (curve_length >= end_curve_length);
    return succeeded;
}

//---------------------------------------------------------------------------//
/*!
 *  Estimate the new trial step for a failed step
 */
CELER_FUNCTION
real_type FieldIntegrator::new_step_size(real_type hstep, real_type error) const
{
    CELER_ASSERT(error > 0);
    real_type scale_factor = (error > 1.0) ? std::pow(error, shared_.pshrink)
                                           : std::pow(error, shared_.pgrow);
    return shared_.safety * hstep * scale_factor;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for find_next_step
 * Check whether the distance to the chord is small than the reference
 */
CELER_FUNCTION bool FieldIntegrator::check_sagitta(real_type       hstep,
                                                   const ode_type& y,
                                                   const ode_type& dydx,
                                                   ode_type&       yend,
                                                   real_type&      dyerr,
                                                   real_type&      dchord)
{
    // Do a quick advance always starting from the initial point
    yend  = y;
    dyerr = this->quick_advance(hstep, yend, dydx, dchord);

    return (dchord <= shared_.delta_chord + rel_tolerance());
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for accurate_advance
 * advance within an allowed step range and
 */
CELER_FUNCTION bool FieldIntegrator::move_step(real_type  h,
                                               real_type  h_threshold,
                                               real_type  end_curve_length,
                                               ode_type&  y,
                                               real_type& hnext,
                                               real_type& curve_length)
{
    // Perform the integration

    real_type hdid{0.};
    ode_type  dydx = stepper_.ode_rhs(y);

    if (h > shared_.minimum_step)
    {
        hdid = this->one_good_step(h, y, dydx, hnext);
    }
    else
    {
        real_type dchord_step; // not used here
        real_type dyerr = this->quick_advance(h, y, dydx, dchord_step);
        hdid            = h;

        // Compute suggested new step
        CELER_ASSERT(h != 0.0);
        hnext = this->new_step_size(h, dyerr / (h * shared_.epsilon_step));
    }

    // Update the current curve length
    curve_length += hdid;

    // Avoid numerous small last steps
    return (h < h_threshold || curve_length >= end_curve_length);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
