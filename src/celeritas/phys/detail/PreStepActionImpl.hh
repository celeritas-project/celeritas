//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/PreStepActionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/random/distribution/ExponentialDistribution.hh"
#include "celeritas/track/SimTrackView.hh"

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

        // Clear step limit and associated action for an empty track slot
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

    // Sample mean free path
    auto phys = track.make_physics_view();
    if (!phys.has_interaction_mfp())
    {
        auto                               rng = track.make_rng_engine();
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    // Calculate physics step limits and total macro xs
    auto      mat      = track.make_material_view();
    auto      particle = track.make_particle_view();
    StepLimit limit    = calc_physics_step_limit(mat, particle, phys, step);
    sim.reset_step_limit(limit);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
