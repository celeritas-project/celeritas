//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldSubstepper.hh
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
template<class IntegratorT>
class FieldSubstepper
{
  public:
    // Construct with options data and the integrator
    inline CELER_FUNCTION FieldSubstepper(FieldDriverOptions const& options,
                                          IntegratorT&& integrate);

    // For a given trial step, advance by a sub_step within a tolerance error
    inline CELER_FUNCTION Substep operator()(real_type step,
                                             OdeState const& state);

    //// TESTABLE HELPER FUNCTIONS ////

    // An adaptive step size control from G4MagIntegratorDriver
    // Move this to private after all tests with non-uniform field are done
    inline CELER_FUNCTION Substep accurate_advance(real_type step,
                                                   OdeState const& state,
                                                   real_type hinitial) const;

    //// ACCESSORS (TODO: refactor) ////

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

    // Integrator for this field driver
    IntegratorT integrate_;

    // Maximum chord length based on a previous estimate
    real_type max_chord_{numeric_limits<real_type>::infinity()};

    //// TYPES ////

    //! A helper output for private member functions
    struct ChordSearch
    {
        Substep end;  //!< Step taken and post-step state
        real_type err_sq;  //!< Square of the truncation error
    };

    struct Integration
    {
        Substep end;  //!< Step taken and post-step state
        real_type proposed_length;  //!< Proposed next step size
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
template<class IntegratorT>
CELER_FUNCTION FieldSubstepper(FieldDriverOptions const&, IntegratorT&&)
    -> FieldSubstepper<IntegratorT>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with options and the step advancement functor.
 */
template<class IntegratorT>
CELER_FUNCTION
FieldSubstepper<IntegratorT>::FieldSubstepper(FieldDriverOptions const& options,
                                              IntegratorT&& integrate)
    : options_(options)
    , integrate_(::celeritas::forward<IntegratorT>(integrate))
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
template<class IntegratorT>
CELER_FUNCTION Substep
FieldSubstepper<IntegratorT>::operator()(real_type step, OdeState const& state)
{
    if (step <= options_.minimum_step)
    {
        // If the input is a very tiny step, do a "quick advance".
        Substep result;
        result.state = integrate_(step, state).end_state;
        result.length = step;
        return result;
    }

    // Calculate the next chord length (and get an end state "for free") based
    // on delta_chord, reusing previous estimates
    ChordSearch next
        = this->find_next_chord(celeritas::min(step, max_chord_), state);
    CELER_ASSERT(next.end.length <= step);
    if (next.end.length < step)
    {
        // Chord length was reduced due to constraints: save the estimate for
        // the next potential field advance inside the propagation loop
        max_chord_ = next.end.length * (1 / options_.min_chord_shrink);
    }

    if (next.err_sq > 1)
    {
        // Discard the original end state and advance more accurately with the
        // newly proposed (reduced) step
        real_type next_step = step * this->new_step_scale(next.err_sq);
        next.end = this->accurate_advance(next.end.length, state, next_step);
    }

    CELER_ENSURE(next.end.length > 0 && next.end.length <= step);
    return next.end;
}

//---------------------------------------------------------------------------//
/*!
 * Find the maximum step length that satisfies a maximum "miss distance".
 */
template<class IntegratorT>
CELER_FUNCTION auto
FieldSubstepper<IntegratorT>::find_next_chord(real_type step,
                                              OdeState const& state) const
    -> ChordSearch
{
    bool succeeded = false;
    auto remaining_steps = options_.max_nsteps;
    FieldIntegration integrated;

    do
    {
        // Try with the proposed step
        integrated = integrate_(step, state);

        // Check whether the distance to the chord is smaller than the
        // reference
        real_type dchord = detail::distance_chord(
            state.pos, integrated.mid_state.pos, integrated.end_state.pos);

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
    ChordSearch result;
    result.end.length = step;
    result.end.state = integrated.end_state;
    result.err_sq = detail::rel_err_sq(integrated.err_state, step, state.mom)
                    / ipow<2>(options_.epsilon_rel_max);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Accurate advance for an adaptive step control.
 *
 * Perform an adaptive step integration for a proposed step or a series of
 * sub-steps within a required tolerance until the the accumulated curved path
 * is equal to the input step length.
 *
 * TODO: maybe this should be moved out of the substepper into the propagation
 * loop?
 */
template<class IntegratorT>
CELER_FUNCTION Substep FieldSubstepper<IntegratorT>::accurate_advance(
    real_type step, OdeState const& state, real_type hinitial) const
{
    CELER_ASSERT(step > 0);

    // Set an initial proposed step and evaluate the minimum threshold
    real_type end_curve_length = step;

    // Use a pre-defined initial step size if it is smaller than the input
    // step length and larger than the per-million fraction of the step length.
    // Otherwise, use the input step length for the first trial.
    // TODO: review whether this approach is an efficient bootstrapping.
    real_type h
        = ((hinitial > options_.initial_step_tol * step) && (hinitial < step))
              ? hinitial
              : step;
    real_type h_threshold = options_.epsilon_step * step;

    // Output with the next good step
    Integration result;
    result.end.state = state;

    // Perform integration
    bool succeeded = false;
    real_type curve_length = 0;
    auto remaining_steps = options_.max_nsteps;

    do
    {
        CELER_ASSERT(h > 0);
        result = this->integrate_step(h, result.end.state);

        curve_length += result.end.length;

        if (h < h_threshold || curve_length >= end_curve_length)
        {
            succeeded = true;
        }
        else
        {
            h = celeritas::min(
                celeritas::max(result.proposed_length, options_.minimum_step),
                end_curve_length - curve_length);
        }
    } while (!succeeded && --remaining_steps > 0);

    // Curve length may be slightly longer than step due to roundoff in
    // accumulation
    CELER_ENSURE(curve_length > 0
                 && (curve_length <= step || soft_equal(curve_length, step)));
    result.end.length = min(curve_length, step);
    return result.end;
}

//---------------------------------------------------------------------------//
/*!
 * Advance for a given step and evaluate the next predicted step.
 *
 * Helper function for accurate_advance.
 */
template<class IntegratorT>
CELER_FUNCTION auto
FieldSubstepper<IntegratorT>::integrate_step(real_type step,
                                             OdeState const& state) const
    -> Integration
{
    CELER_EXPECT(step > 0);

    // Output with a next proposed step
    Integration result;

    if (step > options_.minimum_step)
    {
        result = this->one_good_step(step, state);
    }
    else
    {
        // Do an integration step for a small step (a.k.a quick advance)
        FieldIntegration integrated = integrate_(step, state);

        // Update position and momentum
        result.end.state = integrated.end_state;
        result.end.length = step;

        // Compute a proposed new step
        real_type err_sq
            = detail::rel_err_sq(integrated.err_state, step, state.mom)
              / ipow<2>(options_.epsilon_rel_max);
        result.proposed_length = step * this->new_step_scale(err_sq);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Advance within a relative truncation error and estimate a good step size
 * for the next integration.
 */
template<class IntegratorT>
CELER_FUNCTION auto
FieldSubstepper<IntegratorT>::one_good_step(real_type step,
                                            OdeState const& state) const
    -> Integration
{
    // Perform integration for adaptive step control with the truncation error
    bool succeeded = false;
    size_type remaining_steps = options_.max_nsteps;
    real_type err_sq;
    FieldIntegration integrated;

    do
    {
        integrated = integrate_(step, state);

        err_sq = detail::rel_err_sq(integrated.err_state, step, state.mom)
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
    Integration result;
    result.end.state = integrated.end_state;
    result.end.length = step;
    result.proposed_length
        = step
          * min(this->new_step_scale(err_sq), options_.max_stepping_increase);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Estimate the new predicted step size based on the error estimate.
 */
template<class IntegratorT>
CELER_FUNCTION real_type
FieldSubstepper<IntegratorT>::new_step_scale(real_type err_sq) const
{
    CELER_ASSERT(err_sq >= 0);
    return options_.safety
           * fastpow(err_sq,
                     half() * (err_sq > 1 ? options_.pshrink : options_.pgrow));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
