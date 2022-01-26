//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
 * - The physical (geometry track state) position may deviate from the exact
 *   curved propagation position up to a driver-based tolerance at every
 *   boundary crossing. The momentum will always be conserved, though.
 * - In some unusual cases (e.g. a very small caller-requested step, or an
 *   unusual accumulation in the driver's substeps) the distance returned may
 *   be slightly higher (again, up to a driver-based tolerance) than the
 *   physical distance travelled.
 */
template<class DriverT>
CELER_FUNCTION auto FieldPropagator<DriverT>::operator()(real_type step)
    -> result_type
{
    result_type result;

    // Break the curved steps into substeps as determined by the driver *and*
    // by the proximity of geometry boundaries. Test for intersection with the
    // geometry boundary in each substep. This loop is guaranteed to converge
    // since the trial step always decreases *or* the actual position advances.
    real_type remaining = step;
    while (remaining >= driver_.minimum_step())
    {
        CELER_ASSERT(soft_zero(distance(state_.pos, track_.pos())));

        // Advance up to (but probably less than) the remaining step length
        DriverResult substep = driver_.advance(remaining, state_);
        CELER_ASSERT(substep.step <= remaining);

        // TODO: use safety distance to reduce number of calls to
        // find_next_step

        // Check whether the chord for this sub-step intersects a boundary
        auto chord = detail::make_chord(state_.pos, substep.state.pos);

        // Do a detailed check boundary check from the start position toward
        // the substep end point.
        track_.set_dir(chord.dir);
        real_type linear_step = track_.find_next_step();
        if (linear_step > chord.length)
        {
            // No boundary intersection along the chord: accept substep
            // movement inside the current volume and reset the remaining
            // distance so we can continue toward the next boundary or end of
            // caller-requested step.
            state_ = substep.state;
            result.distance += substep.step;
            remaining = step - result.distance;
            track_.move_internal(state_.pos);
        }
        else if (substep.step * linear_step
                 <= driver_.minimum_step() * chord.length)
        {
            // We're close enough to the boundary that the next trial step
            // would be less than the driver's minimum step. Hop to the
            // boundary without committing the substep.
            result.boundary = true;
            result.distance += linear_step;
            remaining = 0;
        }
        else if (detail::is_intercept_close(state_.pos,
                                            chord.dir,
                                            linear_step,
                                            substep.state.pos,
                                            driver_.delta_intersection()))
        {
            // The straight-line intersection point is a distance less than
            // `delta_intersection` from the substep's end position.
            // Commit the proposed state's momentum but use the
            // post-boundary-crossing track position for consistency
            result.boundary = true;
            result.distance += substep.step;
            state_.mom = substep.state.mom;
            remaining  = 0;
        }
        else
        {
            // The straight-line intercept is too far from substep's end state.
            // Decrease the allowed substep (curved path distance) by the
            // fraction along the chord, and retry the driver step.
            remaining = substep.step * linear_step / chord.length;
        }
    }

    if (result.boundary)
    {
        track_.move_to_boundary();
        state_.pos = track_.pos();
    }
    else if (remaining > 0)
    {
        // Bad luck with substep accumulation or possible very small initial
        // value for "step". Return that we've moved this tiny amount (for e.g.
        // dE/dx purposes) but don't physically propagate the track.
        result.distance += remaining;
    }

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
