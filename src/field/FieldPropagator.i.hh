//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.i.hh
//---------------------------------------------------------------------------//
#include "base/NumericLimits.hh"
#include "detail/FieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared field parameters and the field driver.
 */
template<class DriverT>
CELER_FUNCTION
FieldPropagator<DriverT>::FieldPropagator(const ParticleTrackView& particle,
                                          GeoTrackView*            track,
                                          DriverT*                 driver)
    : track_(*track), driver_(*driver)
{
    CELER_ASSERT(track && driver);
    CELER_ASSERT(particle.charge() != zero_quantity());

    using MomentumUnits = OdeState::MomentumUnits;

    state_.pos = track->pos();
    state_.mom = detail::ax(value_as<MomentumUnits>(particle.momentum()),
                            track->dir());
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle until it hits a boundary.
 */
template<class DriverT>
CELER_FUNCTION auto FieldPropagator<DriverT>::operator()() -> result_type
{
    return (*this)(numeric_limits<real_type>::infinity());
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a magnetic field.
 *
 * It utilises a magnetic field driver based on an adaptive step
 * control to track a charged particle until it travels along a curved
 * trajectory for a given step length within a required accuracy or intersects
 * with a new volume (geometry limited step).
 */
template<class DriverT>
CELER_FUNCTION auto FieldPropagator<DriverT>::operator()(real_type step)
    -> result_type
{
    result_type result;

    // If not a valid range, transportation should not be a candidate process
    if (step < driver_.minimum_step())
    {
        // XXX this should be replaced by a straight-line step or something,
        // including boundary intersection. It should be incorporated into the
        // loop to remove the addition inside the `while` condition.
        result.distance = step;
        result.boundary = false;
        return result;
    }

    // Break the curved steps into substeps as determined by the driver. Test
    // for intersection with the geometry boundary in each substep.
    do
    {
        // Advance up to (but probably less than) the remaining step length
        DriverResult substep = driver_.advance(step - result.distance, state_);

        // Check whether the chord for this sub-step intersects a boundary
        auto chord = detail::make_chord(state_.pos, substep.state.pos);

        real_type safety = track_.find_safety(state_.pos);
        if (chord.length > safety)
        {
            // Potential intersection with boundary (length is less than
            // safety). Do a detailed check boundary check from the start
            // position toward the substep point.
            real_type linear_step
                = track_.compute_step(state_.pos, chord.dir, &safety);

            if (linear_step <= chord.length)
            {
                // We intersect a boundary along the chord.
                Real3 est_intercept_pos = state_.pos;
                axpy(linear_step, chord.dir, &est_intercept_pos);

                unsigned int remaining_steps = driver_.max_nsteps();
                do
                {
                    // Scale the substep (curved path distance) by the fraction
                    // along the chord and save the proposed
                    // along-the-chord intersection point
                    substep.step *= linear_step / chord.length;

                    // Advance from beginning of substep
                    substep.state = state_;
                    substep = driver_.advance(substep.step, substep.state);

                    // Update the intersect candidate point
                    chord = detail::make_chord(state_.pos, substep.state.pos);

                    // Do a detailed boundary check
                    linear_step
                        = track_.compute_step(state_.pos, chord.dir, &safety);

                    est_intercept_pos = state_.pos;
                    axpy(linear_step, chord.dir, &est_intercept_pos);

                    // Check whether substep.state point is within an
                    // acceptable tolerance from the proposed intersect
                    // position on a boundary
                    result.boundary
                        = distance(est_intercept_pos, substep.state.pos)
                          < driver_.delta_intersection();
                    substep.state.pos = est_intercept_pos;
                } while (!result.boundary && --remaining_steps > 0);

                // TODO: loop check and handle rare cases if happen
                CELER_ASSERT(result.boundary);

                // Complete move from start point to intersection
                track_.propagate_state(
                    state_.pos,
                    detail::make_chord(state_.pos, substep.state.pos).dir);
            }
        }

        // Update substep state
        result.distance += substep.step;
        state_ = substep.state;
    } while (!result.boundary
             && (result.distance + driver_.minimum_step()) < step);

    // Update GeoTrackView and return result
    Real3 dir = state_.mom;
    normalize_direction(&dir);
    track_.set_dir(dir);
    track_.set_pos(state_.pos);

    return result;
}
//---------------------------------------------------------------------------//
} // namespace celeritas
