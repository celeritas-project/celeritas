//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldDriver.i.hh
//---------------------------------------------------------------------------//

#include "base/NumericLimits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and the stepper.
 */
CELER_FUNCTION
FieldDriver::FieldDriver(const FieldParamsPointers&           shared,
                         RungeKuttaStepper<MagFieldEquation>& stepper)
    : shared_(shared), stepper_(stepper)
{
    CELER_ENSURE(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step control based on G4ChordFinder and G4MagIntegratorDriver:
 * For a given trial step, advance by a sub-step within a required tolerance
 * and update the current state (position and momentum).  For an efficient
 * adaptive integration, the proposed chord of which the miss-distance (the
 * closest distance from the curved trajectory to the chord) is smaller than
 * a reference distance (dist_chord) will be accepted if its stepping error is
 * within a reference accuracy. Otherwise, the more accurate step integration
 * (advance_accurate) will be performed.
 */
CELER_FUNCTION
real_type FieldDriver::operator()(real_type step, OdeState* state)
{
    // Output with a step control error
    FieldOutput output = this->find_next_chord(step, *state);

    real_type step_taken = output.step_taken;

    // Evaluate the relative error
    real_type rel_error = output.error / (shared_.epsilon_step * step_taken);

    if (rel_error > 1)
    {
        // Advance more accurately with a newly proposed step
        real_type next_step = this->new_step_size(step, rel_error);
        step_taken
            = this->accurate_advance(step_taken, &output.state, next_step);
    }

    // Accept this accuracy and update the current state
    *state = output.state;

    return step_taken;
}

//---------------------------------------------------------------------------//
/*!
 * Find the next acceptable chord of which the miss-distance is smaller than
 * a given reference (delta_chord) and evaluate the associated error.
 */
CELER_FUNCTION auto
FieldDriver::find_next_chord(real_type step, const OdeState& state)
    -> FieldOutput
{
    // Output with a step control error
    FieldOutput output;

    // Try with the proposed step
    output.step_taken = step;

    bool          converged       = false;
    unsigned int  remaining_steps = shared_.max_nsteps;
    StepperResult result;

    do
    {
        result = stepper_(step, state);

        // Check whether the distance to the chord is small than the reference
        real_type dchord
            = distance_chord(state, result.mid_state, result.end_state);

        if (dchord <= (shared_.delta_chord + FieldDriver::rel_tolerance()))
        {
            converged    = true;
            output.error = truncation_error(
                step, shared_.epsilon_rel_max, state, result.err_state);
        }
        else
        {
            // Estimate a new trial chord with a relative scale
            output.step_taken
                *= std::fmax(std::sqrt(shared_.delta_chord / dchord), half());
        }
    } while (!converged && (--remaining_steps > 0));

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(converged);

    // Update new position and momentum
    output.state = result.end_state;

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Accurate_advance for an adaptive step control: Perform an adaptive step
 * integration for a proposed step or a series of sub-steps within a required
 * tolerance until the the accumulated curved path is equal to the input step
 * length.
 */
CELER_FUNCTION real_type FieldDriver::accurate_advance(real_type step,
                                                       OdeState* state,
                                                       real_type hinitial)
{
    CELER_ASSERT(step > 0);

    // Set an initial proposed step and evaluate the minimum threshold
    real_type end_curve_length = step;

    real_type h = ((hinitial > FieldDriver::rel_tolerance() * step)
                   && (hinitial < step))
                      ? hinitial
                      : step;
    real_type h_threshold = shared_.epsilon_step * step;

    // Output with the next good step
    FieldOutput output;

    // Performance integration
    bool         succeeded       = false;
    real_type    curve_length    = 0;
    unsigned int remaining_steps = shared_.max_nsteps;

    do
    {
        output = this->integrate_step(h, *state);

        curve_length += output.step_taken;

        if (h < h_threshold || curve_length >= end_curve_length)
        {
            succeeded = true;
        }
        else
        {
            h = std::fmax(output.next_step, shared_.minimum_step);
            if (curve_length + h > end_curve_length)
            {
                h = end_curve_length - curve_length;
            }
        }
        *state = output.state;
    } while (!succeeded && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(succeeded);

    return curve_length;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for accurate_advance: advance for a given step and
 * evaluate the next predicted step.
 */
CELER_FUNCTION auto
FieldDriver::integrate_step(real_type step, const OdeState& state)
    -> FieldOutput
{
    // Output with a next proposed step
    FieldOutput output;

    if (step > shared_.minimum_step)
    {
        output = this->one_good_step(step, state);
    }
    else
    {
        // Do an integration step for a small step (a.k.a quick advance)
        StepperResult result = stepper_(step, state);

        // Update position and momentum
        output.state = result.end_state;

        real_type dyerr = truncation_error(
            step, shared_.epsilon_rel_max, state, result.err_state);
        output.step_taken = step;

        // Compute a proposed new step
        CELER_ASSERT(output.step_taken != 0);
        output.next_step
            = this->new_step_size(step, dyerr / (step * shared_.epsilon_step));
    }

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Advance within a relative truncation error and estimate a good step size
 * for the next integration.
 */
CELER_FUNCTION auto
FieldDriver::one_good_step(real_type step, const OdeState& state)
    -> FieldOutput
{
    // Output with a proposed next step
    FieldOutput output;

    // Perform integration for adaptive step control with the trunction error
    bool          condition       = false;
    unsigned int  remaining_steps = shared_.max_nsteps;
    real_type     errmax2 = celeritas::numeric_limits<real_type>::max();
    StepperResult result;

    do
    {
        result  = stepper_(step, state);
        errmax2 = truncation_error(
            step, shared_.epsilon_rel_max, state, result.err_state);

        if (errmax2 <= 1)
        {
            condition = true;
        }
        else
        {
            // Step failed; compute the size of re-trial step.
            real_type htemp = shared_.safety * step
                              * std::pow(errmax2, half() * shared_.pshrink);

            // Truncation error too large, reduce stepsize with a low bound
            step = std::fmax(htemp, shared_.max_stepping_decrease * step);
        }
    } while (!condition && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(condition);

    // Update state, step taken by this trial and the next predicted step
    output.state      = result.end_state;
    output.step_taken = step;
    output.next_step  = (errmax2 > ipow<2>(shared_.errcon))
                           ? shared_.safety * step
                                 * std::pow(errmax2, half() * shared_.pgrow)
                           : shared_.max_stepping_increase * step;

    return output;
}

//---------------------------------------------------------------------------//
/*!
 *  Estimate the new predicted step size based on the error estimate
 */
CELER_FUNCTION
real_type FieldDriver::new_step_size(real_type step, real_type rel_error) const
{
    CELER_ASSERT(rel_error > 0);
    real_type scale_factor = (rel_error > 1)
                                 ? std::pow(rel_error, shared_.pshrink)
                                 : std::pow(rel_error, shared_.pgrow);
    return shared_.safety * step * scale_factor;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
