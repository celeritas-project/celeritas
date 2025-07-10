//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Surface.test.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <set>

#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/optical/surface/SurfacePhysicsView.hh"

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
/*!
 * A simple mock for surface models.
 */
template<SurfacePhysicsStep S>
class MockSurfaceModel : public SurfaceModel<S>
{
  public:
    using typename SurfaceModel<S>::CoreStateHost;
    using typename SurfaceModel<S>::CoreStateDevice;

    static typename SurfaceModel<S>::ModelBuilder make_builder()
    {
        return [](ActionId aid) {
            return std::make_shared<MockSurfaceModel<S>>(aid);
        };
    }

    MockSurfaceModel(ActionId id)
        : SurfaceModel<S>(
              id,
              "mock-surface-" + std::to_string(id.get()),
              "mock-surface-description-" + std::to_string(id.get()))
    {
    }

    void step(CoreParams const&, CoreStateHost&) const final {}
    void step(CoreParams const&, CoreStateDevice&) const final {}
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceTest : public OpticalMockTestBase
{
  protected:
    void SetUp() override {}

    SPConstSurfacePhysics build_surface_physics() override
    {
        SurfacePhysicsParams::Input input;

        input.roughness_model_builders.push_back(
            MockSurfaceModel<SurfacePhysicsStep::Roughness>::make_builder());
        input.roughness_model_builders.push_back(
            MockSurfaceModel<SurfacePhysicsStep::Roughness>::make_builder());
        input.roughness_model_builders.push_back(
            MockSurfaceModel<SurfacePhysicsStep::Roughness>::make_builder());

        input.reflectivity_model_builders.push_back(
            MockSurfaceModel<SurfacePhysicsStep::Reflectivity>::make_builder());

        input.interaction_model_builders.push_back(
            MockSurfaceModel<SurfacePhysicsStep::Interaction>::make_builder());
        input.interaction_model_builders.push_back(
            MockSurfaceModel<SurfacePhysicsStep::Interaction>::make_builder());

        auto make_surface = [&input](unsigned int roughness,
                                     unsigned int reflectivity,
                                     unsigned int interaction) {
            input.surfaces.push_back(SurfacePhysicsParams::SurfaceInput{
                RoughnessModelId{roughness},
                ReflectivityModelId{reflectivity},
                InteractionModelId{interaction}});
        };

        make_surface(0, 0, 0);
        make_surface(2, 0, 1);
        make_surface(1, 0, 1);
        make_surface(1, 0, 0);
        make_surface(2, 0, 0);

        input.action_registry = this->action_reg().get();

        return std::make_shared<SurfacePhysicsParams const>(std::move(input));
    }

    void initialize_states(TrackSlotId::size_type num_tracks)
    {
        surface_physics_state_
            = CollectionStateStore<SurfacePhysicsStateData, MemSpace::host>(
                num_tracks);
        CELER_ENSURE(surface_physics_state_.ref().size() == num_tracks);
    }

    SurfacePhysicsView make_surface_view(TrackSlotId slot)
    {
        return SurfacePhysicsView(this->surface_physics()->host_ref(),
                                  surface_physics_state_.ref(),
                                  slot);
    }

  private:
    CollectionStateStore<SurfacePhysicsStateData, MemSpace::host>
        surface_physics_state_;
};

//---------------------------------------------------------------------------//
// Test initialization with trivial surface data
TEST_F(SurfaceTest, trivial_init)
{
    SurfacePhysicsParams::Input input;
    input.action_registry = this->action_reg().get();

    EXPECT_TRUE(std::make_shared<SurfacePhysicsParams const>(std::move(input)));
}

//---------------------------------------------------------------------------//
// Check construction of surface physics parameters from mock data.
TEST_F(SurfaceTest, surface_physics_params)
{
    auto sur_phys = this->surface_physics();
    EXPECT_TRUE(sur_phys->init_boundary_action());

    EXPECT_EQ(3, sur_phys->roughness_models().size());
    EXPECT_EQ(1, sur_phys->reflectivity_models().size());
    EXPECT_EQ(2, sur_phys->interaction_models().size());

    // Check built model metadata

    std::vector<std::string_view> model_names;
    std::vector<std::string_view> model_descs;
    std::set<ActionId> action_ids;

    for (auto const& model : sur_phys->roughness_models())
    {
        ASSERT_TRUE(model);
        model_names.emplace_back(model->label());
        model_descs.emplace_back(model->description());
        action_ids.insert(model->action_id());
    }

    for (auto const& model : sur_phys->reflectivity_models())
    {
        ASSERT_TRUE(model);
        model_names.emplace_back(model->label());
        model_descs.emplace_back(model->description());
        action_ids.insert(model->action_id());
    }

    for (auto const& model : sur_phys->interaction_models())
    {
        ASSERT_TRUE(model);
        model_names.emplace_back(model->label());
        model_descs.emplace_back(model->description());
        action_ids.insert(model->action_id());
    }

    static std::string_view expected_names[] = {
        "mock-surface-2",
        "mock-surface-3",
        "mock-surface-4",
        "mock-surface-5",
        "mock-surface-6",
        "mock-surface-7",
    };
    EXPECT_VEC_EQ(expected_names, model_names);

    static std::string_view expected_descs[] = {
        "mock-surface-description-2",
        "mock-surface-description-3",
        "mock-surface-description-4",
        "mock-surface-description-5",
        "mock-surface-description-6",
        "mock-surface-description-7",
    };
    EXPECT_VEC_EQ(expected_descs, model_descs);

    EXPECT_EQ(6, action_ids.size());
    for (auto action_id : range(ActionId{2}, ActionId{8}))
    {
        EXPECT_EQ(1, action_ids.count(action_id));
    }

    // Check built surface metadata

    auto const& surfaces = sur_phys->host_ref().surfaces;

    EXPECT_EQ(5, surfaces.size());

    std::vector<unsigned int> roughness_models;
    std::vector<unsigned int> reflectivity_models;
    std::vector<unsigned int> interaction_models;

    for (auto sid : range(SurfaceId{surfaces.size()}))
    {
        roughness_models.push_back(surfaces[sid].roughness_model.get());
        reflectivity_models.push_back(surfaces[sid].reflectivity_model.get());
        interaction_models.push_back(surfaces[sid].interaction_model.get());
    }

    static unsigned int expected_roughness_models[] = {2, 4, 3, 3, 4};
    EXPECT_VEC_EQ(expected_roughness_models, roughness_models);

    static unsigned int expected_reflectivity_models[] = {5, 5, 5, 5, 5};
    EXPECT_VEC_EQ(expected_reflectivity_models, reflectivity_models);

    static unsigned int expected_interaction_models[] = {6, 7, 7, 6, 6};
    EXPECT_VEC_EQ(expected_interaction_models, interaction_models);
}

//---------------------------------------------------------------------------//
// Test initializer for surface physics view
TEST_F(SurfaceTest, init_surface_physics_view)
{
    TrackSlotId::size_type num_tracks{10};
    this->initialize_states(num_tracks);

    static unsigned int const expected_surface_ids[]
        = {3, 0, 4, 1, 1, 2, 3, 4, 0, 2};

    static Real3 const expected_surface_normals[] = {
        Real3{1, 0, 0},
        Real3{-1, 0, 0},
        Real3{1, 0, 0},
        Real3{0, 0, 1},
        Real3{1, 0, 0},
        Real3{0, 1, 0},
        Real3{0, 1, 0},
        Real3{1, 0, 0},
        Real3{0, 0, -1},
        Real3{0, -1, 0},
    };

    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        auto surface = this->make_surface_view(tid);
        surface = SurfacePhysicsView::Initializer{
            SurfaceId{expected_surface_ids[tid.get()]},
            expected_surface_normals[tid.get()]};
    }

    std::vector<unsigned int> surface_ids;
    std::vector<unsigned int> roughness_models;
    std::vector<unsigned int> reflectivity_models;
    std::vector<unsigned int> interaction_models;
    std::vector<Real3> surface_normals;
    for (auto tid : range(TrackSlotId{num_tracks}))
    {
        auto const surface = this->make_surface_view(tid);
        surface_ids.push_back(surface.surface_id().get());
        roughness_models.push_back(surface.roughness_action_id().get());
        reflectivity_models.push_back(surface.reflectivity_action_id().get());
        interaction_models.push_back(surface.interaction_action_id().get());
        surface_normals.push_back(surface.surface_normal());
    }

    EXPECT_VEC_EQ(expected_surface_ids, surface_ids);
    EXPECT_VEC_EQ(expected_surface_normals, surface_normals);

    static unsigned int expected_roughness_ids[]
        = {3, 2, 4, 4, 4, 3, 3, 4, 2, 3};
    EXPECT_VEC_EQ(expected_roughness_ids, roughness_models);

    static unsigned int expected_reflectivity_ids[]
        = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    EXPECT_VEC_EQ(expected_reflectivity_ids, reflectivity_models);

    static unsigned int expected_interaction_ids[]
        = {6, 6, 6, 7, 7, 7, 6, 6, 6, 7};
    EXPECT_VEC_EQ(expected_interaction_ids, interaction_models);
}

//---------------------------------------------------------------------------//
// Test surface interaction applier
TEST_F(SurfaceTest, surface_interaction_applier) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
