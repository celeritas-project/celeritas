//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/SoftEqual.hh"

#include "FieldDriverOptions.hh"
#include "Types.hh"

#include "detail/FieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Advance the field state by a single substep based on user tolerances.
 *
 * The substep length is based on the radius of curvature for the step,
 * ensuring that the "miss distance" (sagitta, the distance between the
 * straight-line arc and the furthest point) is less than the \c delta_chord
 * option. This target length is reduced into sub-substeps if necessary to meet
 * a targeted relative error `epsilon_rel_max` based on the position and
 * momentum update.
 *
 * This iteratively reduces the given step length until the sagitta is no more
 * than \c delta_chord . The sagitta is calculated as the projection of the
 * mid-step point onto the line between the start and end-step points.
 *
 * Each iteration reduces the step length by a factor of no more than \c
 * min_chord_shrink , but is based on an approximate "exact" correction factor
 * if the chord length is very small and the curve is circular.
 * The sagitta \em h is related to the chord length \em s and radius of
 * curvature \em r with the trig expression: \f[
   r - h = r \cos \frac{s}{2r}
  \f]
 * For small chord lengths or a large radius, we expand
 * \f$ \cos \theta \sim 1 - \frac{\theta^2}{2} \f$, giving a radius of
 * curvature \f[
   r = \frac{s^2}{8h} \; .
   \f]
 * Given a trial step (chord length) \em s and resulting sagitta of \em h,
 * the exact step needed to give a chord length of \f$ \epsilon = {} \f$ \c
 * delta_chord is \f[
   s' = s \sqrt{\frac{\epsilon}{h}} \,.
 * \f]
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
                                               OdeState const& state);

    // An adaptive step size control from G4MagIntegratorDriver
    // Move this to private after all tests with non-uniform field are done
    inline CELER_FUNCTION DriverResult accurate_advance(
        real_type step, OdeState const& state, real_type hinitial) const;

    //// ACCESSORS ////

    CELER_FUNCTION short int max_substeps() const
    {
        return options_.max_substeps;
    }

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

    // Maximum chord length based on a previous estimate
    real_type max_chord_{numeric_limits<real_type>::infinity()};

    //// TYPES ////

    //! A helper output for private member functions
    struct ChordSearch
    {
        DriverResult end;  //!< Step taken and post-step state
        real_type err_sq;  //!< Square of the truncation error
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
    inline CELER_FUNCTION real_type new_step_scale(real_type error_sq) const;

    //// COMMON PROPERTIES ////

    static CELER_CONSTEXPR_FUNCTION real_type half() { return 0.5; }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class StepperT>
CELER_FUNCTION
FieldDriver(FieldDriverOptions const&, StepperT&&) -> FieldDriver<StepperT>;

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
 * largest distance from the curved trajectory to the chord) is smaller than
 * a reference distance (dist_chord) will be accepted if its stepping error is
 * within a reference accuracy. Otherwise, the more accurate step integration
 * (advance_accurate) will be performed.
 */
template<class StepperT>
CELER_FUNCTION DriverResult
FieldDriver<StepperT>::advance(real_type step, OdeState const& state)
{
    if (step <= options_.minimum_step)
    {
        // If the input is a very tiny step, do a "quick advance".
        DriverResult result;
        result.state = apply_step_(step, state).end_state;
        result.step = step;
        return result;
    }

    // Calculate the next chord length (and get an end state "for free") based
    // on delta_chord, reusing previous estimates
    ChordSearch output
        = this->find_next_chord(celeritas::min(step, max_chord_), state);
    CELER_ASSERT(output.end.step <= step);
    if (output.end.step < step)
    {
        // Chord length was reduced due to constraints: save the estimate for
        // the next potential field advance inside the propagation loop
        max_chord_ = output.end.step * (1 / options_.min_chord_shrink);
    }

    if (output.err_sq > 1)
    {
        // Discard the original end state and advance more accurately with the
        // newly proposed (reduced) step
        real_type next_step = step * this->new_step_scale(output.err_sq);
        output.end = this->accurate_advance(output.end.step, state, next_step);
    }

    CELER_ENSURE(output.end.step > 0 && output.end.step <= step);
    return output.end;
}

//---------------------------------------------------------------------------//
/*!
 * Find the maximum step length that satisfies a maximum "miss distance".
 */
template<class StepperT>
CELER_FUNCTION auto FieldDriver<StepperT>::find_next_chord(
    real_type step, OdeState const& state) const -> ChordSearch
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
            state.pos, result.mid_state.pos, result.end_state.pos);

        if (dchord > options_.delta_chord + options_.dchord_tol)
        {
            // Estimate a new trial chord with a relative scale
            real_type scale_step = max(std::sqrt(options_.delta_chord / dchord),
                                       options_.min_chord_shrink);
            step *= scale_step;
        }
        else
        {
            succeeded = true;
        }
    } while (!succeeded && --remaining_steps > 0);

    // Update step, position and momentum
    output.end.step = step;
    output.end.state = result.end_state;
    output.err_sq = detail::rel_err_sq(result.err_state, step, state.mom)
                    / ipow<2>(options_.epsilon_rel_max);

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
                                      OdeState const& state) const -> Integration
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
        output.end.step = step;

        // Compute a proposed new step
        real_type err_sq = detail::rel_err_sq(result.err_state, step, state.mom)
                           / ipow<2>(options_.epsilon_rel_max);
        output.proposed_step = step * this->new_step_scale(err_sq);
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
FieldDriver<StepperT>::one_good_step(real_type step,
                                     OdeState const& state) const -> Integration
{
    // Output with a proposed next step
    Integration output;

    // Perform integration for adaptive step control with the truncation error
    bool succeeded = false;
    size_type remaining_steps = options_.max_nsteps;
    real_type err_sq;
    FieldStepperResult result;

    do
    {
        result = apply_step_(step, state);

        err_sq = detail::rel_err_sq(result.err_state, step, state.mom)
                 / ipow<2>(options_.epsilon_rel_max);

        if (err_sq > 1)
        {
            // Truncation error too large, reduce stepsize with a low bound
            step *= max(this->new_step_scale(err_sq),
                        options_.max_stepping_decrease);
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
    output.proposed_step
        = step
          * min(this->new_step_scale(err_sq), options_.max_stepping_increase);

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Estimate the new predicted step size based on the error estimate.
 */
template<class StepperT>
CELER_FUNCTION real_type
FieldDriver<StepperT>::new_step_scale(real_type err_sq) const
{
    CELER_ASSERT(err_sq >= 0);
    return options_.safety
           * fastpow(err_sq,
                     half() * (err_sq > 1 ? options_.pshrink : options_.pgrow));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
