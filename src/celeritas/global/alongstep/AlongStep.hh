//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStep.hh
//! \brief Along-step function and helper classes
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the along-step action using helper functions.
 *
 * \tparam MH MSC helper, e.g. \c detail::NoMsc
 * \tparam MP Propagator factory, e.g. \c detail::LinearPropagatorFactory
 * \tparam AE Energy loss applier, e.g. \c detail::NoElossApplier
 */
template<class MH, class MP, class AE>
inline CELER_FUNCTION void along_step(MH&& msc,
                                      MP&& make_propagator,
                                      AE&& apply_eloss,
                                      CoreTrackView const& track)
{
    // TODO: scope the 'views' so that the lifetimes don't overlap between this
    // function and helper class functions
    auto sim = track.make_sim_view();

    // True step is the actual path length traveled by the particle, including
    // within-step MSC
    AlongStepLocalState local;
    local.step_limit = sim.step_limit();
    CELER_ASSERT(local.step_limit);
    if (local.step_limit.step == 0)
    {
        // Track is stopped: no movement or energy loss will happen
        // (could be a stopped positron waiting for annihilation, or a particle
        // waiting to decay?)
        CELER_ASSERT(track.make_particle_view().is_stopped());
        CELER_ASSERT(local.step_limit.action
                     == track.make_physics_view().scalars().discrete_action());
        // Increment the step counter
        sim.increment_num_steps();
        return;
    }

    local.geo_step = local.step_limit.step;
    bool use_msc = msc.is_applicable(track, local.geo_step);
    if (use_msc)
    {
        msc.calc_step(track, &local);
        CELER_ASSERT(local.geo_step > 0);
        CELER_ASSERT(local.step_limit.step >= local.geo_step);
    }

    {
        auto geo = track.make_geo_view();
        auto propagate = make_propagator(track.make_particle_view(), &geo);
        Propagation p = propagate(local.geo_step);
        if (p.boundary)
        {
            // Stopped at a geometry boundary: this is the new step action.
            CELER_ASSERT(p.distance <= local.geo_step);
            CELER_ASSERT(p.distance < local.step_limit.step);
            local.geo_step = p.distance;
            local.step_limit.action = track.boundary_action();
        }
        else if (p.distance < local.geo_step)
        {
            // Some other internal non-boundary geometry limit has been reached
            // (e.g. too many substeps)
            local.geo_step = p.distance;
            local.step_limit.action = track.propagation_limit_action();
        }
    }

    if (use_msc)
    {
        msc.apply_step(track, &local);
    }
    else
    {
        // Step might have been reduced due to geometry boundary
        local.step_limit.step = local.geo_step;
    }

    // Update track's lab-frame time using the beginning-of-step speed
    {
        auto particle = track.make_particle_view();
        CELER_ASSERT(!particle.is_stopped());
        real_type speed = native_value_from(particle.speed());
        CELER_ASSERT(speed >= 0);
        if (speed > 0)
        {
            // For very small energies (< numeric_limits<real_type>::epsilon)
            // the calculated speed can be zero.
            real_type delta_time = local.step_limit.step / speed;
            sim.add_time(delta_time);
        }
    }

    apply_eloss(track, &local.step_limit);
    CELER_ASSERT(local.step_limit.step > 0);
    CELER_ASSERT(local.step_limit.action);

    {
        auto phys = track.make_physics_view();
        if (local.step_limit.action != phys.scalars().discrete_action())
        {
            // Reduce remaining mean free paths to travel. The 'discrete
            // action' case is launched separately and resets the interaction
            // MFP itself.
            auto step = track.make_physics_step_view();
            real_type mfp = phys.interaction_mfp()
                            - local.step_limit.step * step.macro_xs();
            CELER_ASSERT(mfp > 0);
            phys.interaction_mfp(mfp);
        }
    }

    {
        // Override step limit with action/step changes we applied
        sim.force_step_limit(local.step_limit);
        // Increment the step counter
        sim.increment_num_steps();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
