//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

#include "FieldDriverOptions.hh"
#include "Types.hh"
#include "detail/FieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Integrate with and control the quality of the field integration stepper.
 *
 * \note This class is based on G4ChordFinder and G4MagIntegratorDriver.
 */
template<class StepperT>
class FieldDriver
{
  public:
    // Construct with options data and the stepper
    inline CELER_FUNCTION
    FieldDriver(FieldDriverOptions const& options, StepperT&& perform_step);

    // For a given trial step, advance by a sub_step within a tolerance error
    inline CELER_FUNCTION DriverResult advance(real_type step,
                                               OdeState const& state) const;

    // An adaptive step size control from G4MagIntegratorDriver
    // Move this to private after all tests with non-uniform field are done
    inline CELER_FUNCTION DriverResult accurate_advance(
        real_type step, OdeState const& state, real_type hinitial) const;

    //// ACCESSORS ////

    CELER_FUNCTION real_type minimum_step() const
    {
        return options_.minimum_step;
    }

    // TODO: this should be field propagator data
    CELER_FUNCTION real_type delta_intersection() const
    {
        return options_.delta_intersection;
    }

  private:
    //// DATA ////

    // Driver configuration
    FieldDriverOptions const& options_;

    // Stepper for this field driver
    StepperT apply_step_;

    //// TYPES ////

    //! A helper output for private member functions
    struct ChordSearch
    {
        DriverResult end;  //!< Step taken and post-step state
        real_type error;  //!< Stepper error
    };

    struct Integration
    {
        DriverResult end;  //!< Step taken and post-step state
        real_type proposed_step;  //!< Proposed next step size
    };

    //// HEPER FUNCTIONS ////

    // Find the next acceptable chord whose sagitta is less than delta_chord
    inline CELER_FUNCTION ChordSearch
    find_next_chord(real_type step, OdeState const& state) const;

    // Advance for a given step and evaluate the next predicted step.
    inline CELER_FUNCTION Integration
    integrate_step(real_type step, OdeState const& state) const;

    // Advance within the truncated error and estimate a good next step size
    inline CELER_FUNCTION Integration one_good_step(real_type step,
                                                    OdeState const& state) const;

    // Propose a next step size from a given step size and associated error
    inline CELER_FUNCTION real_type new_step_size(real_type step,
                                                  real_type error) const;

    //// COMMON PROPERTIES ////

    static CELER_CONSTEXPR_FUNCTION real_type half() { return 0.5; }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class StepperT>
CELER_FUNCTION FieldDriver(FieldDriverOptions const&, StepperT&&)
    ->FieldDriver<StepperT>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with options and the step advancement functor.
 */
template<class StepperT>
CELER_FUNCTION
FieldDriver<StepperT>::FieldDriver(FieldDriverOptions const& options,
                                   StepperT&& stepper)
    : options_(options), apply_step_(::celeritas::forward<StepperT>(stepper))
{
    CELER_EXPECT(options_);
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step control based on G4ChordFinder and G4MagIntegratorDriver.
 *
 * \param step maximum step length
 * \param state starting state
 * \return substep and updated state
 *
 * For a given trial step, advance by a sub-step within a required tolerance
 * and update the current state (position and momentum).  For an efficient
 * adaptive integration, the proposed chord of which the sagitta (the
 * closest distance from the curved trajectory to the chord) is smaller than
 * a reference distance (dist_chord) will be accepted if its stepping error is
 * within a reference accuracy. Otherwise, the more accurate step integration
 * (advance_accurate) will be performed.
 */
template<class StepperT>
CELER_FUNCTION DriverResult
FieldDriver<StepperT>::advance(real_type step, OdeState const& state) const
{
    if (step <= options_.minimum_step)
    {
        // If the input is a very tiny step, do a "quick advance".
        DriverResult result;
        result.state = apply_step_(step, state).end_state;
        result.step = step;
        return result;
    }

    // Output with a step control error
    ChordSearch output = this->find_next_chord(step, state);
    CELER_ASSERT(output.end.step <= step);

    // Evaluate the relative error
    real_type rel_error = output.error
                          / (options_.epsilon_step * output.end.step);

    if (rel_error > 1)
    {
        // Discard the original end state and advance more accurately with the
        // newly proposed step
        real_type next_step = this->new_step_size(step, rel_error);
        output.end = this->accurate_advance(output.end.step, state, next_step);
    }

    CELER_ENSURE(output.end.step > 0 && output.end.step <= step);
    return output.end;
}

//---------------------------------------------------------------------------//
/*!
 * Find the next acceptable chord of which the sagitta is smaller than
 * a given reference (delta_chord) and evaluate the associated error.
 */
template<class StepperT>
CELER_FUNCTION auto
FieldDriver<StepperT>::find_next_chord(real_type step,
                                       OdeState const& state) const
    -> ChordSearch
{
    // Output with a step control error
    ChordSearch output;

    bool succeeded = false;
    auto remaining_steps = options_.max_nsteps;
    FieldStepperResult result;

    do
    {
        // Try with the proposed step
        result = apply_step_(step, state);

        // Check whether the distance to the chord is smaller than the
        // reference
        real_type dchord = detail::distance_chord(
            state, result.mid_state, result.end_state);

        if (dchord > options_.delta_chord + options_.dchord_tol)
        {
            // Estimate a new trial chord with a relative scale
            step *= max(std::sqrt(options_.delta_chord / dchord), half());
        }
        else
        {
            succeeded = true;
        }
    } while (!succeeded && --remaining_steps > 0);

    // Update step, position and momentum
    output.end.step = step;
    output.end.state = result.end_state;
    output.error = detail::truncation_error(
        step, options_.epsilon_rel_max, state, result.err_state);

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Accurate advance for an adaptive step control.
 *
 * Perform an adaptive step integration for a proposed step or a series of
 * sub-steps within a required tolerance until the the accumulated curved path
 * is equal to the input step length.
 */
template<class StepperT>
CELER_FUNCTION DriverResult FieldDriver<StepperT>::accurate_advance(
    real_type step, OdeState const& state, real_type hinitial) const
{
    CELER_ASSERT(step > 0);

    // Set an initial proposed step and evaluate the minimum threshold
    real_type end_curve_length = step;

    // Use a pre-defined initial step size if it is smaller than the input
    // step length and larger than the permillion fraction of the step length.
    // Otherwise, use the input step length for the first trial.
    // TODO: review whether this approach is an efficient bootstrapping.
    real_type h
        = ((hinitial > options_.initial_step_tol * step) && (hinitial < step))
              ? hinitial
              : step;
    real_type h_threshold = options_.epsilon_step * step;

    // Output with the next good step
    Integration output;
    output.end.state = state;

    // Perform integration
    bool succeeded = false;
    real_type curve_length = 0;
    auto remaining_steps = options_.max_nsteps;

    do
    {
        CELER_ASSERT(h > 0);
        output = this->integrate_step(h, output.end.state);

        curve_length += output.end.step;

        if (h < h_threshold || curve_length >= end_curve_length)
        {
            succeeded = true;
        }
        else
        {
            h = celeritas::min(
                celeritas::max(output.proposed_step, options_.minimum_step),
                end_curve_length - curve_length);
        }
    } while (!succeeded && --remaining_steps > 0);

    // Curve length may be slightly longer than step due to roundoff in
    // accumulation
    CELER_ENSURE(curve_length > 0
                 && (curve_length <= step || soft_equal(curve_length, step)));
    output.end.step = min(curve_length, step);
    return output.end;
}

//---------------------------------------------------------------------------//
/*!
 * Advance for a given step and evaluate the next predicted step.
 *
 * Helper function for accurate_advance.
 */
template<class StepperT>
CELER_FUNCTION auto
FieldDriver<StepperT>::integrate_step(real_type step,
                                      OdeState const& state) const
    -> Integration
{
    CELER_EXPECT(step > 0);

    // Output with a next proposed step
    Integration output;

    if (step > options_.minimum_step)
    {
        output = this->one_good_step(step, state);
    }
    else
    {
        // Do an integration step for a small step (a.k.a quick advance)
        FieldStepperResult result = apply_step_(step, state);

        // Update position and momentum
        output.end.state = result.end_state;

        real_type dyerr = detail::truncation_error(
            step, options_.epsilon_rel_max, state, result.err_state);
        output.end.step = step;

        // Compute a proposed new step
        output.proposed_step = this->new_step_size(
            step, dyerr / (options_.epsilon_step * step));
    }

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Advance within a relative truncation error and estimate a good step size
 * for the next integration.
 */
template<class StepperT>
CELER_FUNCTION auto
FieldDriver<StepperT>::one_good_step(real_type step, OdeState const& state) const
    -> Integration
{
    // Output with a proposed next step
    Integration output;

    // Perform integration for adaptive step control with the trunction error
    bool succeeded = false;
    size_type remaining_steps = options_.max_nsteps;
    real_type errmax2;
    FieldStepperResult result;

    do
    {
        result = apply_step_(step, state);
        errmax2 = detail::truncation_error(
            step, options_.epsilon_rel_max, state, result.err_state);

        if (errmax2 > 1)
        {
            // Step failed; compute the size of re-trial step.
            real_type htemp = options_.safety * step
                              * fastpow(errmax2, half() * options_.pshrink);

            // Truncation error too large, reduce stepsize with a low bound
            step = max(htemp, options_.max_stepping_decrease * step);
        }
        else
        {
            // Success or possibly nan!
            succeeded = true;
        }
    } while (!succeeded && --remaining_steps > 0);

    // Update state, step taken by this trial and the next predicted step
    output.end.state = result.end_state;
    output.end.step = step;
    output.proposed_step = (errmax2 > ipow<2>(options_.errcon))
                               ? options_.safety * step
                                     * fastpow(errmax2, half() * options_.pgrow)
                               : options_.max_stepping_increase * step;

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Estimate the new predicted step size based on the error estimate.
 */
template<class StepperT>
CELER_FUNCTION real_type
FieldDriver<StepperT>::new_step_size(real_type step, real_type rel_error) const
{
    CELER_ASSERT(rel_error >= 0);
    real_type scale_factor = fastpow(
        rel_error, rel_error > 1 ? options_.pshrink : options_.pgrow);
    return options_.safety * step * scale_factor;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
