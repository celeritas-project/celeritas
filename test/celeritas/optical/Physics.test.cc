//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Physics.test.cc
//---------------------------------------------------------------------------//
#include <iostream>
#include <random>

#include "corecel/data/StateDataStore.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/optical/ParticleData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/PhysicsParams.hh"
#include "celeritas/optical/PhysicsStepUtils.hh"
#include "celeritas/optical/PhysicsTrackView.hh"

#include "OpticalMockModels.hh"
#include "OpticalMockTestBase.hh"
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

class OpticalPhysicsTest : public OpticalMockTestBase
{
  protected:
    using RngEngine = ::celeritas::test::DiagnosticRngEngine<std::mt19937>;

    static constexpr ModelId::size_type num_models = 4;

    void SetUp() override { this->initialize_states(1); }

    SPConstOpticalPhysics build_optical_physics() override
    {
        PhysicsParams::Input input;

        for ([[maybe_unused]] auto i : range(num_models))
        {
            input.model_builders.push_back(MockModelBuilder{});
        }

        input.materials = this->optical_material();
        input.action_registry = this->optical_action_reg().get();

        return std::make_shared<PhysicsParams const>(std::move(input));
    }

    PhysicsTrackView
    make_track_view(OptMatId mat, TrackSlotId slot = TrackSlotId{0})
    {
        CELER_EXPECT(mat < this->num_optical_materials());
        return PhysicsTrackView(this->optical_physics()->host_ref(),
                                physics_state_.ref(),
                                mat,
                                slot);
    }

    ParticleTrackView make_particle_view(TrackSlotId slot = TrackSlotId{0})
    {
        return ParticleTrackView(particle_state_.ref(), slot);
    }

    void initialize_states(TrackSlotId::size_type num_tracks)
    {
        particle_state_
            = StateDataStore<ParticleStateData, MemSpace::host>(num_tracks);
        for (auto i : range(TrackSlotId{num_tracks}))
        {
            particle_state_.ref().energy[i] = 3;
        }
        physics_state_
            = StateDataStore<PhysicsStateData, MemSpace::host>(num_tracks);
        CELER_ENSURE(physics_state_.ref().size() == num_tracks);
    }

    /*!
     * Helper function to test different optical materials while iterating
     * over a different ID.
     */
    template<class T>
    OptMatId cycle_material_id(T other_id)
    {
        return OptMatId((2 * other_id.get() + 3)
                        % this->num_optical_materials());
    }

  private:
    SPConstOpticalPhysics optical_physics_;

    StateDataStore<ParticleStateData, MemSpace::host> particle_state_;
    StateDataStore<PhysicsStateData, MemSpace::host> physics_state_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test optical physics parameter accessors.
TEST_F(OpticalPhysicsTest, physics_params)
{
    auto const& params = *this->optical_physics();

    EXPECT_EQ(num_models, params.num_models());

    // Collect built model metadata
    std::vector<std::string_view> model_names;
    std::vector<std::string_view> model_descs;
    std::set<ActionId> action_ids;
    for (auto m_id : range(ModelId{params.num_models()}))
    {
        auto const& model = params.model(m_id);

        ASSERT_TRUE(model);

        model_names.emplace_back(model->label());
        model_descs.emplace_back(model->description());
        action_ids.insert(model->action_id());
    }

    // Check model names
    static std::string_view expected_names[] = {
        "mock-1",
        "mock-2",
        "mock-3",
        "mock-4",
    };
    EXPECT_VEC_EQ(expected_names, model_names);

    // Check model descriptions
    static std::string_view expected_descs[] = {
        "mock-description-1",
        "mock-description-2",
        "mock-description-3",
        "mock-description-4",
    };
    EXPECT_VEC_EQ(expected_descs, model_descs);

    // Check model actions
    EXPECT_EQ(params.num_models(), action_ids.size());
    for (auto action_id : params.model_actions())
    {
        EXPECT_EQ(1, action_ids.count(action_id));
    }
}

//---------------------------------------------------------------------------//
// Test sampling discrete interactions by per model cross sections
TEST_F(OpticalPhysicsTest, TEST_IF_CELERITAS_DOUBLE(select_discrete))
{
    PhysicsTrackView physics = this->make_track_view(OptMatId{3});
    RngEngine rng_engine;

    // Populate XS scratch space used for each model
    physics = PhysicsTrackView::Initializer{};
    real_type total_xs = 0;
    static double const expected_model_xs[] = {0.11893216075412,
                                               0.038415414940508,
                                               0.018644945136445,
                                               0.010997324744179};
    std::vector<real_type> model_xs(num_models, 0);
    for (auto model : range(ModelId{num_models}))
    {
        model_xs[model.get()]
            = physics.calc_xs(model, this->make_particle_view().energy());
        total_xs += model_xs[model.get()];
    }
    physics.macro_xs(total_xs);

    EXPECT_VEC_SOFT_EQ(expected_model_xs, model_xs);

    // Sample actions based on cross sections
    std::vector<ActionId::size_type> actions;
    static ActionId::size_type const expected_actions[]
        = {1, 2, 4, 1, 1, 1, 1, 4, 4, 4};

    for ([[maybe_unused]] auto i : range(10))
    {
        actions.push_back(select_discrete_interaction(
                              this->make_particle_view(), physics, rng_engine)
                              .get());
    }

    EXPECT_VEC_EQ(expected_actions, actions);
}

//---------------------------------------------------------------------------//
// Test expected step limits and calculation cross sections
TEST_F(OpticalPhysicsTest, calc_step_limits)
{
    PhysicsTrackView physics = this->make_track_view(OptMatId{2});
    ParticleTrackView particle = this->make_particle_view();

    static std::vector<real_type> energies{0.1, 1, 5, 10};
    std::vector<std::vector<real_type>> expected_model_xs_per_energy{
        {
            12.006406151030452,
            6.667764385625069,
            4.615748800013053,
            3.5295746115291053,
        },
        {
            1.2006406151030453,
            0.38972461716887974,
            0.19832692355210077,
            0.11789059014280627,
        },
        {
            0.0439036747357096,
            0.01315496492916648,
            0.006228478239695414,
            0.0036181352104175312,
        },
        {
            0.007710727894083951,
            0.002299288122865045,
            0.0010868566672318657,
            0.0006310511934242025,
        },
    };

    physics.interaction_mfp(100);

    for (auto i : range(energies.size()))
    {
        particle.energy(units::MevEnergy{energies[i]});

        auto const& expected_model_xs = expected_model_xs_per_energy[i];
        real_type expected_total_xs = std::accumulate(
            expected_model_xs.begin(), expected_model_xs.end(), real_type{0});

        StepLimit limits = calc_physics_step_limit(particle, physics);

        // Verify step limits
        EXPECT_EQ(physics.discrete_action(), limits.action);
        EXPECT_SOFT_EQ(physics.interaction_mfp(),
                       limits.step * expected_total_xs);

        // Verify cross sections
        EXPECT_SOFT_EQ(expected_total_xs, physics.macro_xs());
    }
}

//---------------------------------------------------------------------------//
// Test model-action accessors of track views
TEST_F(OpticalPhysicsTest, track_view_actions)
{
    // Note that there shouldn't be material or track dependence on the
    // model-action accessors
    PhysicsTrackView physics = this->make_track_view(OptMatId{0});

    // Model-Action mapping

    EXPECT_EQ(num_models, physics.num_models());
    for (auto model : range(ModelId{physics.num_models()}))
    {
        ActionId action = physics.model_to_action(model);
        EXPECT_TRUE(action);
        EXPECT_EQ(model, physics.action_to_model(action));
    }
}

//---------------------------------------------------------------------------//
// Test interaction MFP methods of track view
TEST_F(OpticalPhysicsTest, track_view_interaction_mfp)
{
    TrackSlotId::size_type num_tracks = 10;
    this->initialize_states(num_tracks);

    // There should be track dependence on interaction MFPs
    // Separate mutation and access loops to check independence
    // Note that there shouldn't be material dependence here

    static real_type const expected_interaction_mfps[]
        = {1, 11, 21, 31, 41, 51, 61, 71, 81, 91};

    // Assign interaction MFP
    for (auto track : range(TrackSlotId{num_tracks}))
    {
        auto physics = this->make_track_view(cycle_material_id(track), track);
        physics.interaction_mfp(expected_interaction_mfps[track.get()]);
    }

    std::vector<real_type> interaction_mfps;
    for (auto track : range(TrackSlotId{num_tracks}))
    {
        auto const physics
            = this->make_track_view(cycle_material_id(track + 3), track);
        EXPECT_TRUE(physics.has_interaction_mfp());
        interaction_mfps.push_back(physics.interaction_mfp());
    }

    EXPECT_VEC_EQ(expected_interaction_mfps, interaction_mfps);

    // Reset interaction MFP
    for (auto track : range(TrackSlotId{num_tracks}))
    {
        auto physics
            = this->make_track_view(cycle_material_id(track + 1), track);
        physics.reset_interaction_mfp();
    }

    for (auto track : range(TrackSlotId{num_tracks}))
    {
        auto const physics
            = this->make_track_view(cycle_material_id(track + 5), track);
        EXPECT_FALSE(physics.has_interaction_mfp());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
