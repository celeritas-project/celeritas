//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DiscreteSelectActionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "sim/CoreTrackView.hh"

#include "../PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a physics process before undergoing a collision.
 */
inline CELER_FUNCTION void
discrete_select_track(celeritas::CoreTrackView const& track)
{
    auto sim = track.make_sim_view();

    // Reached the interaction point: sample the process and determine
    // the corresponding action to take.
    auto     particle = track.make_particle_view();
    auto     phys     = track.make_physics_view();
    {
        auto rng = track.make_rng_engine();
        auto action = select_discrete_interaction(particle, phys, rng);
        CELER_ASSERT(action);
        sim.force_step_limit(action);
    }

    // TODO: sample elements here for models that use precalculated elemental
    // cross sections

    // Reset the MFP counter, to be resampled if the track survives the
    // interaction
    phys.reset_interaction_mfp();

    CELER_ENSURE(!phys.has_interaction_mfp());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
