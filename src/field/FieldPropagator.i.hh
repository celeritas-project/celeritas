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
 *
 * The position of the internal OdeState `state_` should be consistent with the
 * geometry `track_`'s position, but the geometry's direction will be a series
 * of "trial" directions that are the chords between the start and end points
 * of a curved substep through the field. At the end of the propagation step,
 * the geometry state's direction is updated based on the actual value of the
 * calculated momentum.
 *
 * Caveats:
 * - Due to boundary fuzziness, the track may translate slightly when it
 *   intersects a boundary.
 * - Due to minimum driver step length, the resulting distance moved may not
 *   exactly add up to the updated track position (TODO: should we do a
 *   "linear" propagation step when this happens?)
 */
template<class DriverT>
CELER_FUNCTION auto FieldPropagator<DriverT>::operator()(real_type step)
    -> result_type
{
    result_type result;

    // Break the curved steps into substeps as determined by the driver *and*
    // by the proximity of geometry boundaries. Test for intersection with the
    // geometry boundary in each substep.
    real_type remaining     = step;
    bool      near_boundary = false;
    while (remaining >= driver_.minimum_step())
    {
        CELER_ASSERT(soft_zero(distance(state_.pos, track_.pos())));

        // Advance up to (but probably less than) the remaining step length
        DriverResult substep = driver_.advance(remaining, state_);

        // TODO: skip additional checking based on available safety distance

        // Check whether the chord for this sub-step intersects a boundary
        auto chord = detail::make_chord(state_.pos, substep.state.pos);

        // Do a detailed check boundary check from the start position toward
        // the substep end point.
        track_.set_dir(chord.dir);
        real_type linear_step = track_.find_next_step();
        if (near_boundary || linear_step <= chord.length)
        {
            // We intersect a boundary along the chord. Calculate the
            // expected straight-line intersection point.
            Real3 est_intercept_pos = state_.pos;
            axpy(linear_step, chord.dir, &est_intercept_pos);

            if (distance(est_intercept_pos, substep.state.pos)
                < driver_.delta_intersection())
            {
                // The substep's end point is within an acceptable tolerance
                // from the chord's boundary intersection. Commit the proposed
                // state's momentum but used the updated track position.
                result.distance += substep.step;
                remaining       = 0;
                result.boundary = true;
                track_.move_across_boundary();
                state_.mom = substep.state.mom;
                state_.pos = track_.pos();
            }
            else
            {
                // Straight-line intersect is too far from substep's end state.
                // Decrease the allowed substep (curved path distance) by the
                // fraction along the chord, and retry the driver step.
                real_type scale = linear_step / chord.length;
                remaining = substep.step * scale;
                if (scale < 0.5)
                {
                    near_boundary = true;
                }
            }
        }
        else
        {
            // No boundary intersection: accept substep movement inside the
            // current volume
            state_ = substep.state;
            result.distance += substep.step;
            remaining = step - result.distance;
            track_.move_internal(state_.pos);
        }
    }

    // Add any additional remaining substep (less than driver minimum)
    // NOTE: this creates a slight inconsistency between the distance traveled
    // and the actual position
    result.distance += remaining;

    // Even though the along-substep movement was through chord lengths,
    // conserve momentum through the field change by updating the final
    // *direction* based on the state's momentum.
    Real3 dir = state_.mom;
    normalize_direction(&dir);
    track_.set_dir(dir);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
