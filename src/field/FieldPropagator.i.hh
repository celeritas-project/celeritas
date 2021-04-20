//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.i.hh
//---------------------------------------------------------------------------//

#include "geometry/detail/VGCompatibility.hh"
#include <VecGeom/navigation/NavigationState.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared field parameters and the field driver.
 */
CELER_FUNCTION
FieldPropagator::FieldPropagator(const FieldParamsPointers& shared,
                                 FieldDriver&               driver)
    : shared_(shared), driver_(driver)
{
    CELER_ASSERT(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a magnetic field and update the field
 * state. It utilises a magnetic field driver based on an adaptive step
 * control to track a charged particle until it travels along a curved
 * trajectory for a given step length within a required accuracy or intersects
 * with a new volume (geometry limited step).
 */
CELER_FUNCTION real_type FieldPropagator::operator()(FieldTrackView* view)
{
    CELER_ASSERT(view && view->charge() != zero_quantity());

    real_type step = view->step();

    // If not a valid range, transportation shouild not be a candiate process
    if (step < shared_.minimum_step)
    {
        return numeric_limits<real_type>::infinity();
    }

    // Initial parameters and states for the field integration
    real_type    step_taken = 0;
    OdeState     end_state  = view->state();
    Intersection intersect;

    do
    {
        OdeState  beg_state = end_state;
        real_type step_left = step - step_taken;

        // Advance within the tolerence error for the remaining step length
        real_type sub_step = driver_(step_left, &end_state);

        // Check whether this sub-step cross a volume boundary
        view->on_boundary(intersect.intersected);
        this->check_intersection(
            view, beg_state.pos, end_state.pos, &intersect);

        // If it is a geometry limited step, find the intersection point
        if (intersect.intersected)
        {
            intersect.scale *= sub_step;
            end_state = this->locate_intersection(view, beg_state, &intersect);

            // Update states for field and navigation
            sub_step      = intersect.step;
            end_state.pos = intersect.pos;
            view->on_boundary(intersect.intersected);
            view->update_vgstates();
        }

        // Add sub-step until there is no remaining step length
        step_taken += sub_step;

    } while (!intersect.intersected
             && (step_taken + shared_.minimum_step) < step);

    // Update the field track view
    view->state(end_state);
    view->step(step_taken);

    return step_taken;
}

//---------------------------------------------------------------------------//
/*!
 * Check whether the final position of the field integration for a given step
 * is inside the current volume or is crossed any boundary of adjacent volumes.
 */
CELER_FUNCTION
void FieldPropagator::check_intersection(FieldTrackView* view,
                                         const Real3     beg_pos,
                                         const Real3     end_pos,
                                         Intersection*   intersect)
{
    intersect->intersected = false;

    Real3 chord = end_pos;
    axpy(real_type(-1.0), beg_pos, &chord);
    Real3 shift = beg_pos;
    axpy(real_type(-1.0), view->origin(), &shift);

    real_type length = norm(chord);
    CELER_ASSERT(length > 0);

    real_type safety = 0;

    real_type shift_mag = norm(shift);

    if (shift_mag < view->safety())
    {
        safety = view->safety() - shift_mag;
    }

    Real3 dir = chord;
    normalize_direction(&dir);

    if (length < safety)
    {
        // Guaranteed inside the current volume and updat the safety
        view->safety(safety);
    }
    else
    {
        // Check whether the linear step length to the next boundary is
        // smaller than the segment to the final position
        real_type linear_step = view->linear_propagator(beg_pos, dir, length);

        intersect->intersected = (linear_step < length);

        intersect->scale = linear_step / length;
        view->origin(beg_pos);

        // If intersect, estimate the candidate intersection point
        if (intersect->intersected)
        {
            intersect->pos = beg_pos;
            axpy(linear_step, dir, &(intersect->pos));
        }

        // Adjust the intersect scale in the case that it is unity
        if (intersect->scale == 1)
            intersect->scale *= FieldPropagator::scale_factor();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the intersection point within a required accuracy using an iterative
 * method and return the final state by the field driver.
 */
CELER_FUNCTION
OdeState FieldPropagator::locate_intersection(FieldTrackView* view,
                                              const OdeState& bed_state,
                                              Intersection*   intersect)
{
    intersect->intersected = false;
    Real3 beg_pos          = bed_state.pos;

    OdeState     end_state;
    unsigned int remaining_steps = shared_.max_nsteps;

    do
    {
        end_state = bed_state;

        real_type step = driver_(intersect->step, &end_state);

        CELER_ASSERT(step == intersect->step);

        // Check whether end_state point is outside the current volume
        Real3 delta = end_state.pos;
        axpy(real_type(-1.0), intersect->pos, &delta);

        intersect->intersected = (norm(delta) < shared_.delta_intersection);

        if (!intersect->intersected)
        {
            // Estimate a new trial step with the updated position of end_state
            real_type trial_step = intersect->step;
            this->check_intersection(view, beg_pos, end_state.pos, intersect);

            intersect->step = trial_step * intersect->scale;
        }
    } while (!intersect->intersected && --remaining_steps > 0);

    // TODO: loop check and handle rare cases if happen
    CELER_ASSERT(intersect->intersected);

    return end_state;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
