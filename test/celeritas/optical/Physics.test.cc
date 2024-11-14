//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Physics.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <iostream>
#include <random>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/ModelBuilder.hh"
#include "celeritas/optical/ParticleData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/PhysicsParams.hh"
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

template<class Functor>
std::vector<std::vector<std::vector<real_type>>>
build_expected_grids(Functor const& f)
{
    ModelId::size_type num_models = 4;
    OpticalMaterialId::size_type num_materials = 7;

    std::vector<std::vector<std::vector<real_type>>> grids;
    grids.reserve(num_models);
    for (auto model : range(ModelId{num_models}))
    {
        std::vector<std::vector<real_type>> model_grids;
        model_grids.reserve(num_materials);
        for (auto mat : range(OpticalMaterialId{num_materials}))
        {
            size_type n = (model.get() + 1) * 10 + mat.get();
            std::vector<real_type> grid;
            grid.reserve(n + 1);
            for (size_type i : range(n + 1))
            {
                grid.push_back(f(i, n));
            }
            model_grids.push_back(std::move(grid));
        }
        grids.push_back(std::move(model_grids));
    }
    return grids;
}

Span<real_type const>
expected_mfp_energy_grid(OpticalMaterialId mat, ModelId model)
{
    static std::vector<std::vector<std::vector<real_type>>> grids;

    if (grids.empty())
    {
        grids = build_expected_grids([](size_type i, size_type n) {
            return 15 * std::log(real_type(i) / n + 1);
        });
    }

    CELER_EXPECT(model < grids.size());
    CELER_EXPECT(mat < grids[model.get()].size());

    return make_span(grids[model.get()][mat.get()]);
}

Span<real_type const>
expected_mfp_value_grid(OpticalMaterialId mat, ModelId model)
{
    static std::vector<std::vector<std::vector<real_type>>> grids;

    if (grids.empty())
    {
        grids = build_expected_grids(
            [](size_type i, size_type) { return i * i; });
    }

    CELER_EXPECT(model < grids.size());
    CELER_EXPECT(mat < grids[model.get()].size());

    return make_span(grids[model.get()][mat.get()]);
}

class MockModel : public Model
{
  public:
    MockModel(ActionId id)
        : Model(id,
                "mock-" + std::to_string(id.get()),
                "mock desc " + std::to_string(id.get()))
    {
    }

    void build_mfps(OpticalMaterialId mat, MfpBuilder& builder) const final
    {
        ModelId model_id{this->action_id().get() - 1};
        builder(expected_mfp_energy_grid(mat, model_id),
                expected_mfp_value_grid(mat, model_id));
    }

    void step(CoreParams const&, CoreStateHost&) const final {}

    void step(CoreParams const&, CoreStateDevice&) const final {}
};

struct MockModelBuilder : public ModelBuilder
{
    SPModel operator()(ActionId id) const override
    {
        return std::make_shared<MockModel>(id);
    }
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DiscreteSelectActionTest : public ::celeritas::test::Test
{
  protected:
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using SPConstPhysics = std::shared_ptr<PhysicsParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;

    ModelId::size_type const num_models = 4;
    OpticalMaterialId::size_type const num_materials = 7;

    void SetUp() override { this->initialize_states(1); }

    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    SPConstPhysics const& physics_params() const
    {
        static SPConstPhysics p = nullptr;
        if (!p)
        {
            p = this->build_physics_params();
        }
        return p;
    }

    SPConstPhysics build_physics_params() const
    {
        PhysicsParams::Input input;

        for ([[maybe_unused]] auto i : range(num_models))
        {
            input.model_builders.push_back(
                std::make_shared<MockModelBuilder const>());
        }

        input.materials = this->material_params();
        input.action_registry = this->action_registry().get();

        return std::make_shared<PhysicsParams const>(std::move(input));
    }

    SPActionRegistry const& action_registry() const
    {
        static SPActionRegistry a = nullptr;
        if (!a)
        {
            a = std::make_shared<ActionRegistry>();
        }
        return a;
    }

    SPConstMaterials const& material_params() const
    {
        static SPConstMaterials m = nullptr;
        if (!m)
        {
            m = this->build_material_params();
        }
        return m;
    }

    SPConstMaterials build_material_params() const
    {
        MaterialParams::Input input;
        ImportPhysicsVector v{ImportPhysicsVectorType::free,
                              std::vector<double>{1, 2},
                              std::vector<double>{3, 4}};

        for (auto mat : range(OpticalMaterialId{num_materials}))
        {
            input.properties.push_back(ImportOpticalProperty{v});
            input.volume_to_mat.push_back(mat);
        }

        return std::make_shared<MaterialParams const>(std::move(input));
    }

    PhysicsTrackView
    make_track_view(OpticalMaterialId mat, TrackSlotId slot = TrackSlotId{0})
    {
        CELER_EXPECT(mat < num_materials);
        return PhysicsTrackView(
            this->physics_params()->host_ref(), physics_state_.ref(), mat, slot);
    }

    PhysicsStepView make_step_view(TrackSlotId slot = TrackSlotId{0})
    {
        return PhysicsStepView(
            this->physics_params()->host_ref(), physics_state_.ref(), slot);
    }

    ParticleTrackView make_particle_view(TrackSlotId slot = TrackSlotId{0})
    {
        return ParticleTrackView(particle_state_.ref(), slot);
    }

    void initialize_states(TrackSlotId::size_type num_tracks)
    {
        particle_state_
            = CollectionStateStore<ParticleStateData, MemSpace::host>(
                num_tracks);
        physics_state_ = CollectionStateStore<PhysicsStateData, MemSpace::host>(
            this->physics_params()->host_ref(), num_tracks);
        CELER_ENSURE(physics_state_.ref().states.size() == num_tracks);
    }

    /*!
     * Helper function to test different optical materials while iterating
     * over a different ID.
     */
    template<class T>
    OpticalMaterialId cycle_material_id(T other_id)
    {
        return OpticalMaterialId{(2 * other_id.get() + 3) % num_materials};
    }

  private:
    RandomEngine rng_;

    CollectionStateStore<ParticleStateData, MemSpace::host> particle_state_;
    CollectionStateStore<PhysicsStateData, MemSpace::host> physics_state_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test sampling discrete interactions by per model cross sections
TEST_F(DiscreteSelectActionTest, select_discrete)
{
    PhysicsTrackView physics = this->make_track_view(OpticalMaterialId{3});
    PhysicsStepView pstep = this->make_step_view();
    auto& rng_engine = this->rng();

    // Populate XS scratch space used for each model
    physics.interaction_mfp() = 0;
    real_type total_xs = 0;
    static real_type model_xs[] = {1.3, 4.7, 2.1, 3.2};
    for (auto model : range(ModelId{num_models}))
    {
        pstep.per_model_xs(model) = model_xs[model.get()];
        total_xs += model_xs[model.get()];
    }
    pstep.macro_xs(total_xs);

    // Sample actions based on cross sections
    std::vector<ActionId::size_type> actions;

    // Can generate expected action IDs from:
    /*
    auto select_expected = make_selector(
        [&](ModelId i) { return model_xs[i.get()]; },
        ModelId{num_models},
        total_xs);
    for ([[maybe_unused]] auto i : range(10))
    {
        actions.push_back(physics.model_to_action(select_expected(rng_engine)).get());
    }
    PRINT_EXPECTED(actions);
     */
    static ActionId::size_type const expected_actions[]
        = {2, 4, 4, 2, 2, 3, 2, 4, 4, 4};

    for ([[maybe_unused]] auto i : range(10))
    {
        actions.push_back(
            select_discrete_interaction(physics, pstep, rng_engine).get());
    }

    EXPECT_VEC_EQ(expected_actions, actions);
}

//---------------------------------------------------------------------------//
// Test expected step limits and calculation cross sections
TEST_F(DiscreteSelectActionTest, calc_step_limits)
{
    PhysicsTrackView physics = this->make_track_view(OpticalMaterialId{2});
    PhysicsStepView pstep = this->make_step_view();
    ParticleTrackView particle = this->make_particle_view();

    static std::vector<real_type> energies{0.1, 1, 5, 10};
    std::vector<std::vector<real_type>> expected_model_xs_per_energy{
        {12.006406151030452,
         6.667764385625069,
         4.615748800013053,
         3.5295746115291053},
        {1.2006406151030453,
         0.38972461716887974,
         0.19832692355210077,
         0.11789059014280627},
        {0.0439036747357096,
         0.01315496492916648,
         0.006228478239695414,
         0.0036181352104175312},
        {0.007710727894083951,
         0.002299288122865045,
         0.0010868566672318657,
         0.0006310511934242025}};

    physics.interaction_mfp() = 100;

    for (auto i : range(energies.size()))
    {
        particle.energy(units::MevEnergy{energies[i]});

        auto const& expected_model_xs = expected_model_xs_per_energy[i];
        real_type expected_total_xs = std::accumulate(
            expected_model_xs.begin(), expected_model_xs.end(), real_type{0});

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
// Test model-action accessors of track views
TEST_F(DiscreteSelectActionTest, track_view_actions)
{
    // Note that there shouldn't be material or track dependence on the
    // model-action accessors
    PhysicsTrackView physics = this->make_track_view(OpticalMaterialId{0});

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
TEST_F(DiscreteSelectActionTest, track_view_interaction_mfp)
{
    TrackSlotId::size_type num_tracks = 10;
    this->initialize_states(num_tracks);

    // There should be track dependence on interaction MFPs
    // Separate mutation and access loops to check independence
    // Note that there shouldn't be material dependence here

    static real_type const expected_interaction_mfps[] = {
        1,
        11,
        21,
        31,
        41,
        51,
        61,
        71,
        81,
        91,
    };

    // Assign interaction MFP
    for (auto track : range(TrackSlotId{num_tracks}))
    {
        auto physics = this->make_track_view(cycle_material_id(track), track);
        physics.interaction_mfp() = expected_interaction_mfps[track.get()];
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
// Test physics step view cross section scratch space
TEST_F(DiscreteSelectActionTest, step_view_xs_scratch)
{
    TrackSlotId::size_type num_tracks = 10;
    this->initialize_states(num_tracks);

    static real_type const expected_per_model_xs[][4] = {{1, 2, 3, 4},
                                                         {5, 6, 7, 8},
                                                         {9, 10, 11, 12},
                                                         {13, 14, 15, 16},
                                                         {17, 18, 19, 20},
                                                         {21, 22, 23, 24},
                                                         {25, 26, 27, 28},
                                                         {29, 30, 31, 32},
                                                         {33, 34, 35, 36},
                                                         {37, 38, 39, 40}};
    static real_type const expected_macro_xs[] = {
        1,
        101,
        201,
        301,
        401,
        501,
        601,
        701,
        801,
        901,
    };

    // Set all of the data
    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        PhysicsStepView pstep = this->make_step_view(track_id);

        for (auto model : range(ModelId{num_models}))
        {
            pstep.per_model_xs(model)
                = expected_per_model_xs[track_id.get()][model.get()];
        }
        pstep.macro_xs(expected_macro_xs[track_id.get()]);
    }

    // Check all of the data
    std::vector<real_type> macro_xs;
    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        PhysicsStepView const pstep = this->make_step_view(track_id);

        std::vector<real_type> model_xs;
        for (auto model : range(ModelId{num_models}))
        {
            model_xs.push_back(pstep.per_model_xs(model));
            EXPECT_EQ(expected_per_model_xs[track_id.get()][model.get()],
                      pstep.per_model_xs(model));
        }
        EXPECT_VEC_EQ(expected_per_model_xs[track_id.get()], model_xs);

        macro_xs.push_back(pstep.macro_xs());
    }
    EXPECT_VEC_EQ(expected_macro_xs, macro_xs);
}

//---------------------------------------------------------------------------//
/* Test MFP grid ID access by track views
 *
 * Valid grid construction tested by \c MfpBuilder tests. Here we just check
 * that grids retrieved by the track view correspond to the expected data.
 */
TEST_F(DiscreteSelectActionTest, track_view_grids)
{
    TrackSlotId::size_type num_tracks = 10;
    this->initialize_states(num_tracks);

    auto const& grids = this->physics_params()->host_ref().grids;
    auto const& reals = this->physics_params()->host_ref().reals;

    for (auto track_id : range(TrackSlotId{num_tracks}))
    {
        for (auto mat_id : range(OpticalMaterialId{num_materials}))
        {
            auto const physics = this->make_track_view(mat_id, track_id);

            for (auto model_id : range(ModelId{physics.num_models()}))
            {
                auto grid_id = physics.mfp_grid(model_id);

                ASSERT_LT(grid_id, grids.size());
                ValueGrid const& grid = grids[grid_id];

                EXPECT_VEC_EQ(expected_mfp_energy_grid(mat_id, model_id),
                              reals[grid.grid]);
                EXPECT_VEC_EQ(expected_mfp_value_grid(mat_id, model_id),
                              reals[grid.value]);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
