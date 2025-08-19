//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysics.test.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"
#include "celeritas/optical/surface/SurfacePhysicsView.hh"

#include "OpticalMockTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{

std::ostream& operator<<(std::ostream& out, SubsurfaceDirection const& d)
{
    switch (d)
    {
        case SubsurfaceDirection::forward:
            out << "forward";
            break;
        case SubsurfaceDirection::reverse:
            out << "reverse";
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
    return out;
}

namespace test
{
using namespace ::celeritas::test;

template<class T>
using SurfaceStepArray = EnumArray<SurfacePhysicsStep, T>;

using ModelSurfaceId = SurfaceModel::ModelSurfaceId;

auto constexpr forward = SubsurfaceDirection::forward;
auto constexpr reverse = SubsurfaceDirection::reverse;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

template<class IdType, class... Args>
std::vector<IdType> as_id_vec(Args... args)
{
    return std::vector<IdType>{IdType(args)...};
}

struct SurfaceResult
{
    std::vector<OptMatId> materials{};
    std::vector<PhysicsSurfaceId> interfaces{};
    SurfaceStepArray<std::vector<ActionId>> actions;
    SurfaceStepArray<std::vector<ModelSurfaceId>> per_model_ids;
};

struct TraceResult
{
    std::vector<SurfaceTrackPosition> position{};
    std::vector<OptMatId> material{};
    std::vector<PhysicsSurfaceId> interface{};
};

TraceResult trace_directions(SurfacePhysicsView& s_physics,
                             std::vector<SubsurfaceDirection> const& directions)
{
    TraceResult result;

    result.position.push_back(s_physics.subsurface_position());
    result.material.push_back(s_physics.subsurface_material());

    for (auto direction : directions)
    {
        result.interface.push_back(s_physics.subsurface_interface(direction));

        s_physics.cross_subsurface_interface(direction);

        result.position.push_back(s_physics.subsurface_position());
        result.material.push_back(s_physics.subsurface_material());
    }

    return result;
}

class MockSurfaceModel : public SurfaceModel
{
  public:
    static SurfaceModel::ModelBuilder
    make_mock_builder(SurfacePhysicsStep step, size_type n)
    {
        return [step, n](ActionId aid) {
            std::stringstream title_string;
            title_string << to_cstring(step) << "-" << n;
            return std::make_shared<MockSurfaceModel>(title_string.str(), aid);
        };
    }

    MockSurfaceModel(std::string const& title, ActionId aid)
        : SurfaceModel(aid, title, "desc-" + title)
    {
    }

    void step(CoreParams const&, CoreStateHost&) const final {}
    void step(CoreParams const&, CoreStateDevice&) const final {}

    VecSurfaceLayer get_surfaces() const final { return {}; }
};

class SurfacePhysicsTest : public OpticalMockTestBase
{
  protected:
    using SPConstSurfacePhysics = std::shared_ptr<SurfacePhysicsParams const>;

    void SetUp() override {}

    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        SurfacePhysicsParams::Input input;
        input.action_reg = this->optical_action_reg().get();
        input.surfaces = {
            {as_id_vec<OptMatId>(0, 3, 1, 2, 1),
             {
                 {SurfaceModelId{1}, SurfaceModelId{0}, SurfaceModelId{3}},
                 {SurfaceModelId{1}, SurfaceModelId{0}, SurfaceModelId{0}},
                 {SurfaceModelId{0}, SurfaceModelId{0}, SurfaceModelId{2}},
                 {SurfaceModelId{1}, SurfaceModelId{0}, SurfaceModelId{1}},
             }},
            {as_id_vec<OptMatId>(0, 2, 1),
             {
                 {SurfaceModelId{0}, SurfaceModelId{0}, SurfaceModelId{1}},
                 {SurfaceModelId{1}, SurfaceModelId{0}, SurfaceModelId{3}},
             }

            },
            {as_id_vec<OptMatId>(0, 1),
             {
                 {SurfaceModelId{1}, SurfaceModelId{0}, SurfaceModelId{2}},
             }},
        };

        SurfaceStepArray<size_type> num_models{2, 1, 4};
        for (auto step : range(SurfacePhysicsStep::size_))
        {
            for (size_type n : range(num_models[step]))
            {
                input.model_builders[step].push_back(
                    MockSurfaceModel::make_mock_builder(step, n));
            }
        }

        return std::make_shared<SurfacePhysicsParams const>(std::move(input));
    }

    void initialize_states(TrackSlotId::size_type num_tracks)
    {
        surface_physics_state_
            = CollectionStateStore<SurfacePhysicsStateData, MemSpace::host>(
                num_tracks);
        CELER_ASSERT(surface_physics_state_.size() == num_tracks);
    }

    SurfacePhysicsView surface_physics_view(TrackSlotId track)
    {
        return SurfacePhysicsView(this->optical_surface_physics()->host_ref(),
                                  surface_physics_state_.ref(),
                                  track);
    }

  private:
    CollectionStateStore<SurfacePhysicsStateData, MemSpace::host>
        surface_physics_state_;
};

//---------------------------------------------------------------------------//
// Test initialization
TEST_F(SurfacePhysicsTest, init_params)
{
    auto params = this->optical_surface_physics();

    ASSERT_TRUE(params);

    // Check boundary actions
    EXPECT_EQ(ActionId{0}, params->init_boundary_action());
    EXPECT_EQ(ActionId{1}, params->post_boundary_action());

    auto const& data = params->host_ref();
    EXPECT_TRUE(data);

    // Gather surface data
    std::vector<SurfaceResult> surfaces(data.surfaces.size());
    for (auto geo_surface : range(GeometricSurfaceId{data.surfaces.size()}))
    {
        auto const& surface_record = data.surfaces[geo_surface];
        EXPECT_TRUE(surface_record);

        auto& surface = surfaces[geo_surface.get()];

        for (auto i : range(SubsurfaceMaterialId{
                 surface_record.subsurface_materials.size()}))
        {
            surface.materials.push_back(
                data.subsurface_materials[surface_record.subsurface_materials[i]]);
        }

        for (auto i : range(SubsurfaceInterfaceId{
                 surface_record.subsurface_interfaces.size()}))
        {
            auto s = data.subsurface_interfaces[surface_record
                                                    .subsurface_interfaces[i]];
            surface.interfaces.push_back(s);

            for (auto step : range(SurfacePhysicsStep::size_))
            {
                SurfaceId temp_id(s.unchecked_get());

                surface.actions[step].push_back(
                    data.model_maps[step].action_ids[temp_id]);
                surface.per_model_ids[step].push_back(
                    data.model_maps[step].model_surface_ids[temp_id]);
            }
        }
    }

    // Check surface data

    std::vector<SurfaceResult> expected{
        // Geometric Surface 0
        // A | D | B | C | B
        //   0   1   2   3
        {
            as_id_vec<OptMatId>(0, 3, 1, 2, 1),
            as_id_vec<PhysicsSurfaceId>(0, 1, 2, 3),
            {
                as_id_vec<ActionId>(1, 1, 0, 1),
                as_id_vec<ActionId>(0, 0, 0, 0),
                as_id_vec<ActionId>(3, 0, 2, 1),
            },
            {
                as_id_vec<ModelSurfaceId>(0, 1, 0, 2),
                as_id_vec<ModelSurfaceId>(0, 1, 2, 3),
                as_id_vec<ModelSurfaceId>(0, 0, 0, 0),
            },
        },
        // Geometric Surface 1
        // A | C | B
        //   4   5
        {
            as_id_vec<OptMatId>(0, 2, 1),
            as_id_vec<PhysicsSurfaceId>(4, 5),
            {
                as_id_vec<ActionId>(0, 1),
                as_id_vec<ActionId>(0, 0),
                as_id_vec<ActionId>(1, 3),
            },
            {
                as_id_vec<ModelSurfaceId>(1, 3),
                as_id_vec<ModelSurfaceId>(4, 5),
                as_id_vec<ModelSurfaceId>(1, 1),
            },
        },
        // Geometric Surface 2
        // A | B
        //   6
        {
            as_id_vec<OptMatId>(0, 1),
            as_id_vec<PhysicsSurfaceId>(6),
            {
                as_id_vec<ActionId>(1),
                as_id_vec<ActionId>(0),
                as_id_vec<ActionId>(2),
            },
            {
                as_id_vec<ModelSurfaceId>(4),
                as_id_vec<ModelSurfaceId>(6),
                as_id_vec<ModelSurfaceId>(1),
            },
        },
    };

    ASSERT_EQ(expected.size(), surfaces.size());
    for (auto i : range(expected.size()))
    {
        auto const& actual_record = surfaces[i];
        auto const& expected_record = expected[i];

        EXPECT_VEC_EQ(expected_record.materials, actual_record.materials);
        EXPECT_VEC_EQ(expected_record.interfaces, actual_record.interfaces);
        for (auto step : range(SurfacePhysicsStep::size_))
        {
            EXPECT_VEC_EQ(expected_record.actions[step],
                          actual_record.actions[step]);
            EXPECT_VEC_EQ(expected_record.per_model_ids[step],
                          actual_record.per_model_ids[step]);
        }
    }

    // Check surface model data

    SurfaceStepArray<std::vector<std::string_view>> expected_model_names;
    expected_model_names[SurfacePhysicsStep::roughness] = {
        "roughness-0",
        "roughness-1",
    };
    expected_model_names[SurfacePhysicsStep::reflectivity] = {
        "reflectivity-0",
    };
    expected_model_names[SurfacePhysicsStep::interaction] = {
        "interaction-0",
        "interaction-1",
        "interaction-2",
        "interaction-3",
    };

    SurfaceStepArray<std::vector<std::string_view>> expected_model_descs;
    expected_model_descs[SurfacePhysicsStep::roughness] = {
        "desc-roughness-0",
        "desc-roughness-1",
    };
    expected_model_descs[SurfacePhysicsStep::reflectivity] = {
        "desc-reflectivity-0",
    };
    expected_model_descs[SurfacePhysicsStep::interaction] = {
        "desc-interaction-0",
        "desc-interaction-1",
        "desc-interaction-2",
        "desc-interaction-3",
    };

    for (auto step : range(SurfacePhysicsStep::size_))
    {
        std::vector<std::string_view> model_names;
        std::vector<std::string_view> model_descs;
        for (auto const& model : params->models(step))
        {
            model_names.push_back(model->label());
            model_descs.push_back(model->description());
        }

        EXPECT_VEC_EQ(expected_model_names[step], model_names);
        EXPECT_VEC_EQ(expected_model_descs[step], model_descs);
    }
}

//---------------------------------------------------------------------------//
// Check initialization of surface physics views
TEST_F(SurfacePhysicsTest, init_surface_physics_view)
{
    auto expected_surfaces = as_id_vec<GeometricSurfaceId>(0, 1, 2, 2, 0, 1, 0);
    std::vector<SubsurfaceDirection> expected_orientations{
        forward,
        forward,
        forward,
        reverse,
        reverse,
        reverse,
        forward,
    };
    std::vector<size_type> expected_num_positions{5, 3, 2, 2, 5, 3, 5};

    auto params = this->optical_surface_physics();
    this->initialize_states(expected_surfaces.size());

    // Initialize tracks
    for (auto track : range(expected_surfaces.size()))
    {
        this->surface_physics_view(TrackSlotId(track))
            = SurfacePhysicsView::Initializer{expected_surfaces[track],
                                              expected_orientations[track]};
    }

    // Check initialization
    std::vector<GeometricSurfaceId> surfaces;
    std::vector<SubsurfaceDirection> orientations;
    std::vector<size_type> num_positions;
    for (auto track : range(TrackSlotId(expected_surfaces.size())))
    {
        auto s_physics = this->surface_physics_view(track);

        surfaces.push_back(s_physics.surface());
        orientations.push_back(s_physics.orientation());
        num_positions.push_back(s_physics.num_positions());

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());
        EXPECT_EQ(0, s_physics.subsurface_position().get());
    }

    EXPECT_VEC_EQ(expected_surfaces, surfaces);
    EXPECT_VEC_EQ(expected_orientations, orientations);
    EXPECT_VEC_EQ(expected_num_positions, num_positions);

    // Check position in post-volume
    for (auto track : range(TrackSlotId(expected_surfaces.size())))
    {
        auto s_physics = this->surface_physics_view(track);
        s_physics.subsurface_position()
            = SurfaceTrackPosition(s_physics.num_positions() - 1);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_FALSE(s_physics.in_pre_volume());
        EXPECT_TRUE(s_physics.in_post_volume());
        EXPECT_EQ(expected_num_positions[track.get()] - 1,
                  s_physics.subsurface_position().get());
    }

    // Check some intermediate positions
    std::vector<SurfaceTrackPosition> expected_intermediate_positions{
        SurfaceTrackPosition{2},
        SurfaceTrackPosition{1},
        SurfaceTrackPosition{},
        SurfaceTrackPosition{},
        SurfaceTrackPosition{3},
        SurfaceTrackPosition{1},
        SurfaceTrackPosition{1},
    };

    for (auto track : range(TrackSlotId(expected_surfaces.size())))
    {
        if (auto pos = expected_intermediate_positions[track.get()])
        {
            auto s_physics = this->surface_physics_view(track);
            s_physics.subsurface_position() = pos;

            EXPECT_TRUE(s_physics.is_crossing_boundary());
            EXPECT_FALSE(s_physics.in_pre_volume());
            EXPECT_FALSE(s_physics.in_post_volume());
            EXPECT_EQ(pos, s_physics.subsurface_position());
        }
    }

    // Check resetting tracks clears relevant state
    for (auto track : range(TrackSlotId(expected_surfaces.size())))
    {
        auto s_physics = this->surface_physics_view(track);
        s_physics.reset();

        EXPECT_FALSE(s_physics.is_crossing_boundary());
    }
}

//---------------------------------------------------------------------------//
// Check surface view traversing subsurface materials and interfaces
TEST_F(SurfacePhysicsTest, traverse_subsurface)
{
    auto params = this->optical_surface_physics();
    this->initialize_states(10);

    {
        // Geometric surface 2 (forward): A | B
        // Path: A -> B
        std::vector<SubsurfaceDirection> directions{
            forward,
        };

        auto s_physics = this->surface_physics_view(TrackSlotId{0});
        s_physics
            = SurfacePhysicsView::Initializer{GeometricSurfaceId{2}, forward};

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        auto result = trace_directions(s_physics, directions);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_FALSE(s_physics.in_pre_volume());
        EXPECT_TRUE(s_physics.in_post_volume());

        TraceResult expected{as_id_vec<SurfaceTrackPosition>(0, 1),
                             as_id_vec<OptMatId>(0, 1),
                             as_id_vec<PhysicsSurfaceId>(6)};

        EXPECT_VEC_EQ(expected.position, result.position);
        EXPECT_VEC_EQ(expected.material, result.material);
        EXPECT_VEC_EQ(expected.interface, result.interface);
    }
    {
        // Geometric surface 2 (reverse): B | A
        // Path: B -> A
        std::vector<SubsurfaceDirection> directions{
            forward,
        };

        auto s_physics = this->surface_physics_view(TrackSlotId{1});
        s_physics
            = SurfacePhysicsView::Initializer{GeometricSurfaceId{2}, reverse};

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        auto result = trace_directions(s_physics, directions);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_FALSE(s_physics.in_pre_volume());
        EXPECT_TRUE(s_physics.in_post_volume());

        TraceResult expected{as_id_vec<SurfaceTrackPosition>(0, 1),
                             as_id_vec<OptMatId>(1, 0),
                             as_id_vec<PhysicsSurfaceId>(6)};

        EXPECT_VEC_EQ(expected.position, result.position);
        EXPECT_VEC_EQ(expected.material, result.material);
        EXPECT_VEC_EQ(expected.interface, result.interface);
    }
    {
        // Geometric surface 0 (forward): A | D | B' | C | B
        // Path: A -> D -> B' -> D -> B' -> C -> B -> C -> B' -> D -> A
        std::vector<SubsurfaceDirection> directions{
            forward,
            forward,
            reverse,
            forward,
            forward,
            forward,
            reverse,
            reverse,
            reverse,
        };

        auto s_physics = this->surface_physics_view(TrackSlotId{2});
        s_physics
            = SurfacePhysicsView::Initializer{GeometricSurfaceId{0}, forward};

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        auto result = trace_directions(s_physics, directions);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        TraceResult expected{
            as_id_vec<SurfaceTrackPosition>(0, 1, 2, 1, 2, 3, 4, 3, 2, 1, 0),
            as_id_vec<OptMatId>(0, 3, 1, 3, 1, 2, 1, 2, 1, 3, 0),
            as_id_vec<PhysicsSurfaceId>(0, 1, 1, 1, 2, 3, 3, 2, 1, 0)};

        EXPECT_VEC_EQ(expected.position, result.position);
        EXPECT_VEC_EQ(expected.material, result.material);
        EXPECT_VEC_EQ(expected.interface, result.interface);
    }
    {
        // Geometric surface 1 (reverse): B | C | A
        // Path: B -> C -> A -> C -> B -> C -> A
        std::vector<SubsurfaceDirection> directions{
            forward,
            forward,
            reverse,
            reverse,
            forward,
            forward,
        };

        auto s_physics = this->surface_physics_view(TrackSlotId{3});
        s_physics
            = SurfacePhysicsView::Initializer{GeometricSurfaceId{1}, reverse};

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        auto result = trace_directions(s_physics, directions);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_FALSE(s_physics.in_pre_volume());
        EXPECT_TRUE(s_physics.in_post_volume());

        TraceResult expected{
            as_id_vec<SurfaceTrackPosition>(0, 1, 2, 1, 0, 1, 2),
            as_id_vec<OptMatId>(1, 2, 0, 2, 1, 2, 0),
            as_id_vec<PhysicsSurfaceId>(5, 4, 4, 5, 5, 4)};

        EXPECT_VEC_EQ(expected.position, result.position);
        EXPECT_VEC_EQ(expected.material, result.material);
        EXPECT_VEC_EQ(expected.interface, result.interface);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
