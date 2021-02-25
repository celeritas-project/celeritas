//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldIntegrator.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and the stepper.
 */
CELER_FUNCTION
FieldIntegrator::FieldIntegrator(const FieldParamsPointers& shared,
                                 RungeKutta&                stepper)
    : shared_(shared), stepper_(stepper)
{
    CELER_ENSURE(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step control
 */
CELER_FUNCTION
real_type FieldIntegrator::advance_chord_limited(real_type hstep, ode_type& y)
{
    ode_type  yend         = y;
    real_type curve_length = 0;

    real_type dyerr;
    real_type h = find_next_chord(hstep, y, yend, dyerr);

    // evaluate a relative error
    real_type ratio = dyerr / (shared_.epsilon_step * h);

    if (ratio < 1.0)
    {
        // Accept this accuracy and update the state
        y = yend;
    }
    else
    {
        // Compute the hnext for accuracy_advance
        real_type hnext = new_step_size(h, ratio);

        // Advance more accurately to the "end of chord"
        bool is_good = accurate_advance(h, y, curve_length, hnext);
        if (!is_good)
        {
            h = curve_length;
        }
    }
    return h;
}

//---------------------------------------------------------------------------//
/*!
 * find the next chord and return a step length taken and updates the state
 */
CELER_FUNCTION
real_type FieldIntegrator::find_next_chord(real_type       hstep,
                                           const ode_type& y,
                                           ode_type&       yend,
                                           real_type&      dyerr)
{
    // Try with the proposed step
    real_type h = hstep;

    ode_type dydx;
    ode_rhs(y, dydx);

    for (CELER_MAYBE_UNUSED int i : celeritas::range(shared_.max_nsteps))
    {
        // Always start from the initial point
        yend = y;
        real_type dchord_step;

        dyerr = quick_advance(h, yend, dydx, dchord_step);

        // Exit if the distance to the chord is small than the reference
        if (dchord_step <= shared_.delta_chord)
            break;

        // try a reduced step size, but not more than a factor 2
        h *= std::fmax(sqrt(shared_.delta_chord / dchord_step), 0.5);
    }
    //! XXX: loop check for rare failed cases?

    return h;
}

//---------------------------------------------------------------------------//
/*!
 * Adaptive step control
 */
CELER_FUNCTION real_type FieldIntegrator::one_good_step(real_type       h,
                                                        ode_type&       y,
                                                        const ode_type& dydx,
                                                        real_type&      hnext)
{
    real_type errmax2 = 0;
    ode_type  yout;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(shared_.max_nsteps))
    {
        errmax2 = stepper_.stepper(h, y, dydx, yout);
        if (errmax2 <= 1.0)
        {
            break;
        } // step succeeded.

        // Step failed; compute the size of retrial step.
        real_type htemp = shared_.safety * h
                          * std::pow(errmax2, 0.5 * shared_.pshrink);

        // Truncation error too large, reduce stepsize with a low bound
        h = std::fmax(htemp, shared_.max_stepping_decrease * h);
    }

    // Compute size of the next step
    if (errmax2 > shared_.errcon * shared_.errcon)
    {
        hnext = shared_.safety * h * std::pow(errmax2, 0.5 * shared_.pgrow);
    }
    else
    {
        hnext = shared_.max_stepping_increase * h;
    }

    // Update position and momentum and return step taken by this trial
    y = yout;

    return h;
}

CELER_FUNCTION
real_type FieldIntegrator::quick_advance(real_type       h,
                                         ode_type&       y,
                                         const ode_type& dydx,
                                         real_type&      dchord_step)
{
    ode_type yout;

    // Do an integration step
    real_type dyerr = stepper_.stepper(h, y, dydx, yout);

    // Estimate the curve-chord distance
    dchord_step = stepper_.distance_chord(y, yout);

    // Update new position and momentum and return the error of this step
    y = yout;

    return dyerr;
}

CELER_FUNCTION bool FieldIntegrator::accurate_advance(real_type  hstep,
                                                      ode_type&  y,
                                                      real_type& curve_length,
                                                      real_type  hinitial)
{
    // Check validity and initialization

    CELER_ASSERT(hstep > 0);
    bool succeeded = true;

    real_type end_curve_length = curve_length + hstep;

    // set an initial proposed step and evaluate the minimum threshold
    real_type h = ((hinitial > permillion() * hstep) && (hinitial < hstep))
                      ? hinitial
                      : hstep;

    real_type h_threshold = shared_.epsilon_step * hstep;

    // Perform the integration
    real_type hnext, hdid;

    for (CELER_MAYBE_UNUSED int nstep : celeritas::range(shared_.max_nsteps))
    {
        ode_type dydx;
        stepper_.ode_rhs(y, dydx);

        if (h > shared_.minimun_step)
        {
            hdid = one_good_step(h, y, dydx, hnext);
        }
        else
        {
            real_type dchord_step; // not used here
            real_type dyerr = quick_advance(h, y, dydx, dchord_step);
            hdid            = h;

            // Compute suggested new step
            CELER_ASSERT(h != 0.0);
            hnext = new_step_size(h, dyerr / (h * shared_.epsilon_step));
        }

        // Update the current curve length
        curve_length += hdid;

        // Avoid numerous small last steps
        if (h < h_threshold || curve_length >= end_curve_length)
        {
            break;
        }

        h = std::fmax(hnext, shared_.minimun_step);
        if (curve_length + h > end_curve_length)
        {
            h = end_curve_length - curve_length;
        }
    }

    succeeded = (curve_length >= end_curve_length);
    return succeeded;
}

CELER_FUNCTION
real_type FieldIntegrator::new_step_size(real_type hstep, real_type error)
{
    CELER_ASSERT(error > 0);

    // Estimate the new trial step for a failed step
    real_type scale_factor = (error > 1.0) ? std::pow(error, shared_.pshrink)
                                           : std::pow(error, shared_.pgrow);
    return shared_.safety * hstep * scale_factor;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
