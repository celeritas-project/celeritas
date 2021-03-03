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
 * Construct with shared parameters and the field integator.
 */
CELER_FUNCTION
FieldPropagator::FieldPropagator(const FieldParamsPointers& shared,
                                 FieldIntegrator&           integrator)
    : shared_(shared), integrator_(integrator)
{
    CELER_ASSERT(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Propagation in a field
 */
CELER_FUNCTION real_type FieldPropagator::operator()(FieldTrackView& view)
{
    CELER_ASSERT(view.charge().value() != 0);

    real_type step_length = view.h();

    if (step_length < this->tolerance())
    {
        return numeric_limits<real_type>::infinity();
    }

    // Initial parameters for adaptive Runge-Kutta integration
    ode_type  yc = view.y();
    bool      intersect{false};
    real_type step_taken{0.0};
    vec3_type intersect_point{0, 0, 0};

    do
    {
        ode_type  yo    = yc;
        real_type hstep = step_length - step_taken;

        // For a given trial step (hstep), advance a sub_step within the
        // required tolerence error and update current states (yc)
        real_type sub_step = integrator_(hstep, yc);

        // Check whether this sub-step cross a volume boundary
        real_type intersect_scale;
        intersect = this->is_boundary_crossing(view,
                                               detail::to_vector(yo.position()),
                                               detail::to_vector(yc.position()),
                                               intersect_point,
                                               intersect_scale);

        if (intersect)
        {
            // Find the intersection point
            real_type trial_step = intersect_scale * sub_step;
            bool      is_found   = this->find_intersect_point(
                view, yo, yc, intersect_point, trial_step);

            CELER_ASSERT(is_found);
            // A new geometry limited sub-setp
            sub_step = trial_step;
            view.update_vgstates();
            for (int i = 0; i < 3; ++i)
            {
                yc[i] = intersect_point[i];
            }
        }

        step_taken += sub_step;

    } while (!(intersect) && step_taken + tolerance() < step_length);

    // Update the field track view
    view.y() = yc;
    view.h() = step_taken;

    return step_taken;
}

//---------------------------------------------------------------------------//
/*!
 * Check whether there is a boundary crossing by the linear line between
 * two space points
 */
bool FieldPropagator::is_boundary_crossing(FieldTrackView& view,
                                           const vec3_type x_start,
                                           const vec3_type x_end,
                                           vec3_type&      intersect_point,
                                           real_type&      intersect_scale)
{
    vec3_type chord = x_end - x_start;
    vec3_type shift = x_start - view.origin();

    real_type length = chord.Mag();
    CELER_ASSERT(length > 0);

    real_type current_safety = 0.0;

    if (shift.Mag() < view.safety())
    {
        current_safety = view.safety() - shift.Mag();
    }

    bool      is_intersect = false;
    vec3_type dir          = chord.Unit();

    if (length < current_safety)
    {
        // guaranteed with no intersection
        view.safety() = current_safety;
    }
    else
    {
        real_type linear_step = view.linear_propagator(x_start, dir, length);

        is_intersect = (linear_step < length);

        intersect_scale = linear_step / length;
        view.origin()   = x_start;

        if (is_intersect)
        {
            intersect_point = x_start + linear_step * dir;
        }

        // Adjust the trial scale if it is same as before
        if (intersect_scale == 1.0)
        {
            intersect_scale *= 1.01; // XXX optimize
        }
    }

    return is_intersect;
}

//---------------------------------------------------------------------------//
/*!
 * Find the first intersection point on a volume boundary along the curved
 * path in a magnetic field
 */
bool FieldPropagator::find_intersect_point(FieldTrackView& view,
                                           const ode_type& y_start,
                                           ode_type&       y_end,
                                           vec3_type&      intersect_point,
                                           real_type&      trial_step)
{
    vec3_type start_position = detail::to_vector(y_start.position());

    ode_type yt   = y_start;
    bool is_found = check_intersect(yt, y_end, intersect_point, trial_step);

    unsigned int remaining_steps = shared_.max_nsteps;
    while (!is_found && --remaining_steps > 0)
    {
        // Estimate a new trial step with the new position at yt
        real_type new_scale{1.0};

        this->is_boundary_crossing(view,
                                   start_position,
                                   detail::to_vector(yt.position()),
                                   intersect_point,
                                   new_scale);

        trial_step *= new_scale;

        yt       = y_start;
        is_found = check_intersect(yt, y_end, intersect_point, trial_step);
    }
    // XXX TODO: loop check and handle rare cases if happen
    CELER_ASSERT(is_found);

    return is_found;
}

//---------------------------------------------------------------------------//
/*!
 * Helfer for find_intersect_point: move by a new trial step and check whether
 * the new position is on a volume boundary within a required accuracy
 */
bool FieldPropagator::check_intersect(ode_type&  yt,
                                      ode_type&  y_end,
                                      vec3_type& intersect_point,
                                      real_type& trial_step)
{
    bool is_found = false;

    real_type hstep = integrator_(trial_step, yt);

    CELER_ASSERT(hstep == trial_step);

    // Check yt point
    real_type delta
        = (detail::to_vector(yt.position()) - intersect_point).Mag();

    if (delta < shared_.delta_intersection)
    {
        is_found = true;
        y_end    = yt;
    }

    return is_found;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
