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

    // If not a valid range, transportation shouild not be a candidate process
    if (step < driver_.minimum_step())
    {
        result.distance = step;
        result.boundary = false;
        return result;
    }

    // Initial parameters and states for the field integration
    real_type    step_taken = 0;
    Intersection intersect;

    do
    {
        OdeState  beg_state = state_;
        real_type step_left = step - step_taken;

        // Advance within the tolerance error for the remaining step length
        real_type sub_step = driver_.advance(step_left, &state_);

        // Check whether this sub-step intersects with a volume boundary
        result.boundary = intersect.intersected;
        this->query_intersection(beg_state.pos, state_.pos, &intersect);

        // If it is a geometry limited step, find the intersection point
        if (intersect.intersected)
        {
            intersect.step = sub_step * intersect.scale;
            state_         = this->find_intersection(beg_state, &intersect);
            sub_step       = intersect.step;
            state_.pos     = intersect.pos;
            result.boundary = intersect.intersected;

            // Calculate direction from begin_state to current position
            Real3 intersect_dir
                = detail::make_direction(beg_state.pos, state_.pos);
            track_.propagate_state(beg_state.pos, intersect_dir);
        }

        // Add sub-step until there is no remaining step length
        step_taken += sub_step;

    } while (!intersect.intersected
             && (step_taken + driver_.minimum_step()) < step);

    result.distance = step_taken;

    // Update GeoTrackView and return result
    Real3 dir = state_.mom;
    normalize_direction(&dir);
    track_.set_dir(dir);
    track_.set_pos(state_.pos);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Check whether the final position of the field integration for a given step
 * is inside the current volume or beyond any boundary of adjacent volumes.
 */
template<class DriverT>
CELER_FUNCTION void
FieldPropagator<DriverT>::query_intersection(const Real3&  beg_pos,
                                             const Real3&  end_pos,
                                             Intersection* intersect)
{
    intersect->intersected = false;

    Real3 chord;
    for (size_type i = 0; i != 3; ++i)
    {
        chord[i] = end_pos[i] - beg_pos[i];
    }

    real_type length = norm(chord);
    CELER_ASSERT(length > 0);

    real_type safety = track_.find_safety(beg_pos);
    if (length > safety)
    {
        // Check whether the linear step length to the next boundary is
        // smaller than the segment to the final position
        normalize_direction(&chord);

        real_type linear_step = track_.compute_step(beg_pos, chord, &safety);

        intersect->intersected = (linear_step <= length);
        intersect->scale       = linear_step / length;

        // If intersects, estimate the candidate intersection point
        if (intersect->intersected)
        {
            intersect->pos = beg_pos;
            axpy(linear_step, chord, &(intersect->pos));
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the intersection point within a required accuracy using an iterative
 * method and return the final state by the field driver.
 */
template<class DriverT>
CELER_FUNCTION OdeState FieldPropagator<DriverT>::find_intersection(
    const OdeState& beg_state, Intersection* intersect)
{
    intersect->intersected = false;
    Real3 beg_pos          = beg_state.pos;

    OdeState     end_state;
    unsigned int remaining_steps = driver_.max_nsteps();

    do
    {
        end_state = beg_state;

        real_type step = driver_.advance(intersect->step, &end_state);
        CELER_ASSERT(step == intersect->step);

        // Update the intersect candidate point
        Real3 dir = detail::make_direction(beg_pos, end_state.pos);

        real_type safety      = 0;
        real_type linear_step = track_.compute_step(beg_pos, dir, &safety);
        intersect->pos        = beg_pos;
        axpy(linear_step, dir, &intersect->pos);

        // Check whether end_state point is within an acceptable tolerance
        // from the proposed intersect position on a boundary
        intersect->intersected = (distance(intersect->pos, end_state.pos)
                                  < driver_.delta_intersection());

        if (!intersect->intersected)
        {
            // Estimate a new trial step with the updated position of end_state
            real_type trial_step = intersect->step;

            real_type length = distance(beg_pos, end_state.pos);
            CELER_ASSERT(length > 0);

            intersect->scale = (linear_step / length);
            intersect->step  = trial_step * intersect->scale;
        }
    } while (!intersect->intersected && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(intersect->intersected);

    return end_state;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
