//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared field parameters and the field driver.
 */
CELER_FUNCTION
FieldPropagator::FieldPropagator(GeoTrackView&            track,
                                 const ParticleTrackView& particle,
                                 FieldDriver&             driver)
    : track_(track), driver_(driver)
{
    CELER_ASSERT(particle.charge() != zero_quantity());

    state_.pos = track.pos();
    axpy(particle.momentum().value(), track.dir(), &state_.mom);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a magnetic field and update the field
 * state. It utilises a magnetic field driver based on an adaptive step
 * control to track a charged particle until it travels along a curved
 * trajectory for a given step length within a required accuracy or intersects
 * with a new volume (geometry limited step).
 */
CELER_FUNCTION auto FieldPropagator::operator()(real_type step) -> result_type
{
    result_type result;

    // If not a valid range, transportation shouild not be a candiate process
    if (step < driver_.minimum_step())
    {
        result.distance = numeric_limits<real_type>::infinity();
        return result;
    }

    // Initial parameters and states for the field integration
    real_type    step_taken = 0;
    Intersection intersect;

    do
    {
      //        OdeState  beg_state = end_state;
        OdeState  beg_state = state_;
        real_type step_left = step - step_taken;

        // Advance within the tolerance error for the remaining step length
        real_type sub_step = driver_(step_left, &state_);

        // Check whether this sub-step intersects with a volume boundary
        result.on_boundary = intersect.intersected;
        this->query_intersection(beg_state.pos, state_.pos, &intersect);

        // If it is a geometry limited step, find the intersection point
        if (intersect.intersected)
        {
            intersect.step = sub_step * intersect.scale;
            state_      = this->find_intersection(beg_state, &intersect);
            sub_step    = intersect.step;
            state_.pos  = intersect.pos;

            result.on_boundary = intersect.intersected;

            Real3 dir = state_.pos;
            axpy(real_type(-1.0), beg_state.pos, &dir);
            normalize_direction(&dir);
            track_.propagate_state(beg_state.pos, dir);
        }

        // Add sub-step until there is no remaining step length
        step_taken += sub_step;

    } while (!intersect.intersected
             && (step_taken + driver_.minimum_step()) < step);

    result.state    = state_;
    result.distance = step_taken;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Check whether the final position of the field integration for a given step
 * is inside the current volume or beyond any boundary of adjacent volumes.
 */
CELER_FUNCTION
void FieldPropagator::query_intersection(const Real3&  beg_pos,
                                         const Real3&  end_pos,
                                         Intersection* intersect)
{
    intersect->intersected = false;

    Real3 chord = end_pos;
    axpy(real_type(-1.0), beg_pos, &chord);

    real_type length = norm(chord);
    CELER_ASSERT(length > 0);

    real_type safety = track_.find_safety(beg_pos);
    if (length > safety)
    {
        // Check whether the linear step length to the next boundary is
        // smaller than the segment to the final position
        Real3 dir = chord;
        normalize_direction(&dir);

        real_type linear_step = track_.compute_step(beg_pos, dir, &safety);

        intersect->intersected = (linear_step <= length);
        intersect->scale       = linear_step / length;

        // If intersects, estimate the candidate intersection point
        if (intersect->intersected)
        {
            intersect->pos = beg_pos;
            axpy(linear_step, dir, &(intersect->pos));
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the intersection point within a required accuracy using an iterative
 * method and return the final state by the field driver.
 */
CELER_FUNCTION
OdeState FieldPropagator::find_intersection(const OdeState& beg_state,
                                            Intersection*   intersect)
{
    intersect->intersected = false;
    Real3 beg_pos          = beg_state.pos;

    OdeState     end_state;
    unsigned int remaining_steps = driver_.max_nsteps();

    do
    {
        end_state = beg_state;

        real_type step = driver_(intersect->step, &end_state);
        CELER_ASSERT(step == intersect->step);

        // Check whether end_state point is within an acceptable tolerance
        // from the proposed intersect position on a boundary
        Real3 delta = end_state.pos;
        axpy(real_type(-1.0), intersect->pos, &delta);

        intersect->intersected = (norm(delta) < driver_.delta_intersection());

        if (!intersect->intersected)
        {
            // Estimate a new trial step with the updated position of end_state
            real_type trial_step = intersect->step;

            Real3 dir = end_state.pos;
            axpy(real_type(-1.0), beg_pos, &dir);
            normalize_direction(&dir);
            real_type safety      = 0;
            real_type linear_step = track_.compute_step(beg_pos, dir, &safety);

            intersect->scale = (linear_step / intersect->step);
            intersect->step  = trial_step * intersect->scale;

            intersect->pos = beg_pos;
            axpy(linear_step, dir, &intersect->pos);
        }
    } while (!intersect->intersected && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(intersect->intersected);

    return end_state;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
