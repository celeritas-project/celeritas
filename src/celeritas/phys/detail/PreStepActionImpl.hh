//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/PreStepActionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Macros.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/random/distribution/ExponentialDistribution.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../PhysicsStepUtils.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Set up the beginning of a physics step.
 *
 * - Reset track properties (todo: move to track initialization?)
 * - Sample the mean free path and calculate the physics step limits.
 */
inline CELER_FUNCTION void pre_step_track(celeritas::CoreTrackView const& track)
{
    if (track.thread_id() == ThreadId{0})
    {
        // Clear secondary storage on a single thread
        auto alloc = track.make_physics_step_view().make_secondary_allocator();
        alloc.clear();
    }

    auto sim = track.make_sim_view();
    if (sim.status() == TrackStatus::inactive)
    {
#if CELERITAS_DEBUG
        auto step = track.make_physics_step_view();
        step.reset_energy_deposition_debug();
        step.secondaries({});
#endif

        // Clear step limit and actions for an empty track slot
        sim.reset_step_limit();
        return;
    }

    auto step = track.make_physics_step_view();
    {
        // Clear out energy deposition, secondary pointers, and sampled element
        step.reset_energy_deposition();
        step.secondaries({});
        step.element({});
    }

    auto phys = track.make_physics_view();
    if (phys.num_particle_processes() == 0)
    {
        // Replicate G4Transportation
        // Set MFP to infinity and set action as geo-boundary
        sim.reset_step_limit(
            {numeric_limits<real_type>::max(), track.boundary_action()});
        phys.interaction_mfp(numeric_limits<real_type>::max());
        sim.along_step_action()
            = track.core_scalars().along_step_neutral_action;
        return;
    }

    if (!phys.has_interaction_mfp())
    {
        // Sample mean free path
        auto rng = track.make_rng_engine();
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    // Calculate physics step limits and total macro xs
    auto mat = track.make_material_view();
    auto particle = track.make_particle_view();
    sim.reset_step_limit(calc_physics_step_limit(mat, particle, phys, step));

    // Initialize along-step action based on particle charge:
    // This should eventually be dependent on region, energy, etc.
    sim.along_step_action() = [&particle, &scalars = track.core_scalars()] {
        if (particle.charge() == zero_quantity())
        {
            return scalars.along_step_neutral_action;
        }
        else
        {
            return scalars.along_step_user_action;
        }
    }();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
