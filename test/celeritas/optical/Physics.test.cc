//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Physics.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <random>

#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/optical/ParticleData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/PhysicsStepUtils.hh"
#include "celeritas/optical/PhysicsStepView.hh"
#include "celeritas/optical/PhysicsTrackView.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DiscreteSelectActionTest : public ::celeritas::test::Test
{
  protected:
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    void SetUp() override
    {
        particle_state_
            = CollectionStateStore<ParticleStateData, MemSpace::host>(1);
    }

    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    CollectionStateStore<ParticleStateData, MemSpace::host> particle_state_;

  private:
    RandomEngine rng_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test sampling discrete interactions by per model cross sections
TEST_F(DiscreteSelectActionTest, select_discrete)
{
    PhysicsTrackView physics(OpticalMaterialId{0}, TrackSlotId{0});
    PhysicsStepView pstep(TrackSlotId{0});
    auto& rng_engine = this->rng();

    physics.interaction_mfp() = 0;
    real_type total_xs = 0;
    static real_type model_xs[] = {1.3, 4.7, 2.1, 3.2};
    for (auto model : range(ModelId{4}))
    {
        pstep.per_model_xs(model) = model_xs[model.get()];
        total_xs += model_xs[model.get()];
    }
    pstep.macro_xs(total_xs);

    std::vector<ActionId::size_type> actions;
    for ([[maybe_unused]] auto i : range(10))
    {
        actions.push_back(
            select_discrete_interaction(physics, pstep, rng_engine).get());
    }

    static ActionId::size_type expected_actions[] = {0};

    EXPECT_VEC_EQ(expected_actions, actions);
}

//---------------------------------------------------------------------------//
// Test expected step limits and calculation cross sections
TEST_F(DiscreteSelectActionTest, calc_step_limits)
{
    PhysicsTrackView physics(OpticalMaterialId{0}, TrackSlotId{0});
    PhysicsStepView pstep(TrackSlotId{0});
    ParticleTrackView particle(particle_state_.ref(), TrackSlotId(0));

    static std::vector<real_type> energies{0.1, 1, 10};
    static std::vector<std::vector<real_type>> expected_model_xs_per_energy{
        {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    physics.interaction_mfp() = 100;

    for (auto i : range(energies.size()))
    {
        particle.energy(units::MevEnergy{energies[i]});

        auto const& expected_model_xs = expected_model_xs_per_energy[i];
        real_type expected_total_xs = std::accumulate(
            expected_model_xs.begin(), expected_model_xs.end(), 0);

        StepLimit limits = calc_physics_step_limit(particle, physics, pstep);

        // Verify step limits
        EXPECT_EQ(physics.discrete_action(), limits.action);
        EXPECT_SOFT_EQ(physics.interaction_mfp(),
                       limits.step * expected_total_xs);

        // Verify cross sections
        for (auto mid : range(ModelId{physics.num_models()}))
        {
            EXPECT_SOFT_EQ(expected_model_xs[mid.get()],
                           pstep.per_model_xs(mid));
        }
        EXPECT_SOFT_EQ(expected_total_xs, pstep.macro_xs());
    }
}

//---------------------------------------------------------------------------//
// Test physics track view utilities
TEST_F(DiscreteSelectActionTest, track_view)
{
    PhysicsTrackView physics(OpticalMaterialId{0}, TrackSlotId{0});

    // Interaction MFP

    physics.interaction_mfp() = 10;
    EXPECT_TRUE(physics.has_interaction_mfp());
    EXPECT_EQ(10, physics.interaction_mfp());

    physics.reset_interaction_mfp();
    EXPECT_FALSE(physics.has_interaction_mfp());

    // Model-Action mapping

    EXPECT_EQ(4, physics.num_models());
    for (auto model : range(ModelId{physics.num_models()}))
    {
        ActionId action = physics.model_to_action(model);
        EXPECT_TRUE(action);
        EXPECT_EQ(model, physics.action_to_model(action));
    }

    // Access MFP grids

    std::vector<ValueGridId::size_type> mfp_grids;
    for (auto model : range(ModelId{physics.num_models()}))
    {
        mfp_grids.push_back(physics.mfp_grid(model).get());
    }

    static ValueGridId::size_type expected_mfp_grids[] = {0};

    EXPECT_VEC_EQ(expected_mfp_grids, mfp_grids);
}

//---------------------------------------------------------------------------//
// Test physics track view over multiple tracks
TEST_F(DiscreteSelectActionTest, many_track_view)
{
    TrackSlotId::size_type num_tracks = 10;

    // Set all of the data
    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        PhysicsTrackView physics(OpticalMaterialId{0}, track_id);

        physics.interaction_mfp() = static_cast<real_type>(track_id.get());
    }

    // Check all of the data
    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        PhysicsTrackView const physics(OpticalMaterialId{0}, track_id);

        EXPECT_SOFT_EQ(static_cast<real_type>(track_id.get()),
                       physics.interaction_mfp());
    }
}

//---------------------------------------------------------------------------//
// Test physics step view over multiple tracks
TEST_F(DiscreteSelectActionTest, many_step_view)
{
    TrackSlotId::size_type num_tracks = 10;

    // Set all of the data
    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        PhysicsTrackView const physics(OpticalMaterialId{0}, track_id);

        PhysicsStepView pstep(track_id);

        for (auto model : range(ModelId{physics.num_models()}))
        {
            pstep.per_model_xs(model) = track_id.get() * physics.num_models()
                                        + model.get();
        }
        pstep.macro_xs(100 * track_id.get());
    }

    // Check all of the data
    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        PhysicsTrackView const physics(OpticalMaterialId{0}, track_id);

        PhysicsStepView const pstep(track_id);

        for (auto model : range(ModelId{physics.num_models()}))
        {
            EXPECT_SOFT_EQ(
                static_cast<real_type>(track_id.get() * physics.num_models()
                                       + model.get()),
                pstep.per_model_xs(model));
        }
        EXPECT_SOFT_EQ(real_type{100} * track_id.get(), pstep.macro_xs());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
