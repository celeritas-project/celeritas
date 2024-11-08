//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsStepUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/random/Selector.hh"

#include "ParticleTrackView.hh"
#include "PhysicsStepView.hh"
#include "PhysicsTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

inline CELER_FUNCTION StepLimit
calc_physics_step_limit(ParticleTrackView const& particle,
                        PhysicsTrackView const& physics,
                        PhysicsStepView& pstep)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    real_type total_xs = 0;
    for (auto model : range(ModelId{physics.num_models()}))
    {
        real_type model_xs = 1 / physics.calc_mfp(model, particle.energy());
        total_xs += model_xs;
        pstep.per_model_xs(model) = model_xs;
    }
    pstep.macro_xs(total_xs);

    CELER_ASSERT(pstep.macro_xs() > 0);

    StepLimit limit;
    limit.action = physics.discrete_action();
    limit.step = physics.interaction_mfp() / total_xs;

    return limit;
}

template<class Engine>
CELER_FUNCTION ActionId select_discrete_interaction(
    PhysicsTrackView const& physics, PhysicsStepView const& pstep, Engine& rng)
{
    // Should be called after discrete select action has reset the MFP and the
    // macroscopic cross sections have bene built.
    CELER_EXPECT(!physics.has_interaction_mfp());
    CELER_EXPECT(pstep.macro_xs() > 0);

    ModelId mid = celeritas::make_selector(
        [&pstep](ModelId m) { return pstep.per_model_xs(m); },
        ModelId{physics.num_models()},
        pstep.macro_xs())(rng);

    return physics.model_to_action(mid);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
