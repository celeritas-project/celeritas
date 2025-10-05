//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsStepUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/random/distribution/Selector.hh"

#include "ParticleTrackView.hh"
#include "PhysicsTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the discrete physics step limit for the given track.
 *
 * The total cross section is cached in the \c PhysicsTrackView.
 */
inline CELER_FUNCTION StepLimit calc_physics_step_limit(
    ParticleTrackView const& particle, PhysicsTrackView& physics)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    real_type total_xs = 0;
    for (auto model : range(ModelId{physics.num_models()}))
    {
        total_xs += physics.calc_xs(model, particle.energy());
    }
    physics.macro_xs(total_xs);

    CELER_ASSERT(physics.macro_xs() > 0);

    StepLimit limit;
    limit.action = physics.discrete_action();
    limit.step = physics.interaction_mfp() / total_xs;

    return limit;
}

//---------------------------------------------------------------------------//
/*!
 * Randomly sample a discrete interaction by their cross sections.
 *
 * Should be performed after discrete select action has reset the MFP,
 * and the macroscopic cross sections have been built.
 */
template<class Engine>
CELER_FUNCTION ActionId
select_discrete_interaction(ParticleTrackView const& particle,
                            PhysicsTrackView const& physics,
                            Engine& rng)
{
    CELER_EXPECT(!physics.has_interaction_mfp());
    CELER_EXPECT(physics.macro_xs() > 0);

    ModelId mid = celeritas::make_selector(
        [&physics, energy = particle.energy()](ModelId m) {
            return physics.calc_xs(m, energy);
        },
        ModelId{physics.num_models()},
        physics.macro_xs())(rng);

    return physics.model_to_action(mid);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
