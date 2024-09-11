//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsStepUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/random/Selector.hh"

#include "PhysicsStepView.hh"
#include "PhysicsTrackView.hh"
#include "TrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

inline CELER_FUNCTION StepLimit calc_physics_step_limit(
    TrackView const& particle, PhysicsTrackView& physics, PhysicsStepView& pstep)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    real_type total_macro_xs = 0;
    for (auto mid : range(ModelId{physics.num_optical_models()}))
    {
        real_type model_xs = 1 / physics.calc_mfp(mid, particle.energy());
        total_macro_xs += model_xs;
        pstep.per_model_xs(mid) = model_xs;
    }
    pstep.macro_xs(total_macro_xs);

    CELER_ASSERT(total_macro_xs > 0);

    StepLimit limit;
    limit.action = physics.scalars().discrete_action();
    limit.step = physics.interaction_mfp() / total_macro_xs;

    return limit;
}

template<class Engine>
CELER_FUNCTION ActionId select_discrete_interaction(
    PhysicsTrackView const& physics, PhysicsStepView& pstep, Engine& rng)
{
    // Should be called after discrete select action has reset the MFP
    // and the macroscopic cross sections have been built
    CELER_EXPECT(physics.interaction_mfp() <= 0);
    CELER_EXPECT(pstep.macro_xs() > 0);

    ModelId mid = celeritas::make_selector(
        [&pstep](ModelId mid) { return pstep.per_model_xs(mid); },
        ModelId{physics.num_optical_models()},
        pstep.macro_xs())(rng);

    return physics.model_to_action(mid);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
