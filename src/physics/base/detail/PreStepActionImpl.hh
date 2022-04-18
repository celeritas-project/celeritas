//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PreStepLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "sim/CoreTrackView.hh"
#include "sim/Types.hh"

#include "../PhysicsStepUtils.hh"

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
 * - TODO: add user fixed step limit
 */
inline CELER_FUNCTION void pre_step_track(celeritas::CoreTrackView const& track)
{
    if (track.thread_id() == ThreadId{0})
    {
        // Clear secondary storage on a single thread
        auto alloc = track.make_secondary_allocator();
        alloc.clear();
    }


    auto sim = track.make_sim_view();
    if (sim.status() == TrackStatus::inactive)
    {
#if CELERITAS_DEBUG
        auto phys = track.make_physics_view_inactive();
        phys.reset_energy_deposition_debug();
        phys.secondaries({});
#endif

        // Clear step limit and associated action for an empty track slot
        sim.reset_step_limit();
        return;
    }

    auto phys = track.make_physics_view();
    {
        // Clear out energy deposition and secondary pointers
        phys.reset_energy_deposition();
        phys.secondaries({});
    }

    // Sample mean free path
    if (!phys.has_interaction_mfp())
    {
        auto                               rng = track.make_rng_engine();
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    // Calculate physics step limits and total macro xs
    auto      mat      = track.make_material_view();
    auto      particle = track.make_particle_view();
    StepLimit limit    = calc_physics_step_limit(mat, particle, phys);
    sim.reset_step_limit(limit);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
