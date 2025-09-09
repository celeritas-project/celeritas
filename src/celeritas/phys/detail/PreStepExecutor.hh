//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/PreStepExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../PhysicsStepUtils.hh"  // IWYU pragma: associated
#include "../PhysicsStepView.hh"
#include "../PhysicsTrackView.hh"

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
 *
 * \note This executor applies to *all* tracks, including inactive ones. It
 *   \em must be run on all thread IDs to properly initialize secondaries.
 */
struct PreStepExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
PreStepExecutor::operator()(celeritas::CoreTrackView const& track)
{
    if (track.thread_id() == ThreadId{0})
    {
        // Clear secondary storage on a single thread
        auto alloc = track.physics_step().make_secondary_allocator();
        alloc.clear();
    }

    auto sim = track.sim();
    if (sim.status() == TrackStatus::inactive)
    {
#if CELERITAS_DEBUG
        auto step = track.physics_step();
        step.reset_energy_deposition_debug();
        step.secondaries({});
#endif

        // Clear step limit and actions for an empty track slot
        sim.reset_step_limit();
        return;
    }

    // Complete the "initializing" stage of tracks, since pre-step happens
    // after user initialization
    auto step = track.physics_step();
    {
        // Clear out energy deposition, secondary pointers, and sampled element
        step.reset_energy_deposition();
        step.secondaries({});
        step.element({});
    }

    if (CELER_UNLIKELY(sim.status() == TrackStatus::errored))
    {
        // Failed during initialization: don't calculate step limits
        return;
    }

    CELER_ASSERT(sim.status() == TrackStatus::initializing
                 || sim.status() == TrackStatus::alive);
    sim.status(TrackStatus::alive);

    auto phys = track.physics();
    if (!phys.has_interaction_mfp())
    {
        // Sample mean free path
        auto rng = track.rng();
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    // Calculate physics step limits and total macro xs
    auto mat = track.material();
    auto particle = track.particle();
    sim.reset_step_limit(calc_physics_step_limit(mat, particle, phys, step));

    // Initialize along-step action based on particle charge:
    // This should eventually be dependent on region, energy, etc.
    sim.along_step_action([&particle, &scalars = track.core_scalars()] {
        if (particle.charge() == zero_quantity())
        {
            return scalars.along_step_neutral_action;
        }
        else
        {
            return scalars.along_step_user_action;
        }
    }());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
