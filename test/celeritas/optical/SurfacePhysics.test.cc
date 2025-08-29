//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysics.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/inp/SurfacePhysics.hh"

#include <memory>
#include <set>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/VariantUtils.hh"
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
using SurfaceOrderArray = EnumArray<SurfacePhysicsOrder, T>;

using InternalSurfaceId = ::celeritas::SurfaceModel::InternalSurfaceId;

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
    std::vector<PhysSurfaceId> interfaces{};
    SurfaceOrderArray<std::vector<SurfaceModelId>> actions;
    SurfaceOrderArray<std::vector<InternalSurfaceId>> per_model_ids;
};

struct TraceResult
{
    std::vector<SurfaceTrackPosition> position{};
    std::vector<OptMatId> material{};
    std::vector<PhysSurfaceId> interface{};
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

class SurfacePhysicsTest : public OpticalMockTestBase
{
  protected:
    using SPConstSurfacePhysics = std::shared_ptr<SurfacePhysicsParams const>;

    void SetUp() override {}

    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        using namespace celeritas::inp;
        using PSI = PhysSurfaceId;

        SurfacePhysics input;
        input.materials = {
            as_id_vec<OptMatId>(3, 1, 2),
            as_id_vec<OptMatId>(2),
            as_id_vec<OptMatId>(),
            as_id_vec<OptMatId>(),
        };

        input.roughness = RoughnessModels{
            {
                {PSI{0}, NoRoughness{}},
                {PSI{1}, NoRoughness{}},
                {PSI{6}, NoRoughness{}},
                {PSI{7}, NoRoughness{}},
            },
            {
                {PSI{2}, SmearRoughness{0.3}},
                {PSI{5}, SmearRoughness{0.7}},
            },
            {
                {PSI{3}, GaussianRoughness{0.07}},
                {PSI{4}, GaussianRoughness{0.13}},
            },
        };

        input.reflectivity = ReflectivityModels{
            {
                {PSI{0}, GridReflection{Grid{{0.0, 1.0}, {0.1, 0.3}}}},
                {PSI{2}, GridReflection{Grid{{0.0, 1.0}, {0.4, 0.5}}}},
                {PSI{5}, GridReflection{Grid{{0.0, 1.0}, {0.2, 0.9}}}},
            },
            {
                {PSI{1}, FresnelReflection{}},
                {PSI{3}, FresnelReflection{}},
                {PSI{4}, FresnelReflection{}},
                {PSI{6}, FresnelReflection{}},
                {PSI{7}, FresnelReflection{}},
            },
        };

        input.interaction = InteractionModels{
            {
                {PSI{0}, ReflectionForm::from_spike()},
                {PSI{3}, ReflectionForm::from_lobe()},
                {PSI{4}, ReflectionForm::from_lambertian()},
                {PSI{6}, ReflectionForm::from_lobe()},
                {PSI{7}, ReflectionForm::from_spike()},
            },
            {
                {PSI{1}, ReflectionForm::from_lambertian()},
                {PSI{2}, ReflectionForm::from_spike()},
                {PSI{5}, ReflectionForm::from_lobe()},
            },
        };

        return std::make_shared<SurfacePhysicsParams const>(
            this->optical_action_reg().get(), input);
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
    EXPECT_EQ(ActionId{1}, params->surface_stepping_action());
    EXPECT_EQ(ActionId{2}, params->post_boundary_action());

    auto const& data = params->host_ref();
    EXPECT_TRUE(data);

    // Gather surface data
    std::vector<SurfaceResult> surfaces(data.surfaces.size());
    for (auto geo_surface : range(SurfaceId{data.surfaces.size()}))
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
            auto phys_surface = surface_record.subsurface_interfaces[i];
            surface.interfaces.push_back(phys_surface);

            for (auto step : range(SurfacePhysicsOrder::size_))
            {
                surface.actions[step].push_back(
                    data.model_maps[step].surface_models[phys_surface]);
                surface.per_model_ids[step].push_back(
                    data.model_maps[step].internal_surface_ids[phys_surface]);
            }
        }
    }

    // Check surface data

    std::vector<SurfaceResult> expected{
        // Geometric Surface 0
        // A | D | B | C | B
        //   0   1   2   3
        {
            as_id_vec<OptMatId>(3, 1, 2),
            as_id_vec<PhysSurfaceId>(0, 1, 2, 3),
            {
                as_id_vec<SurfaceModelId>(0, 0, 1, 2),
                as_id_vec<SurfaceModelId>(0, 1, 0, 1),
                as_id_vec<SurfaceModelId>(0, 1, 1, 0),
            },
            {
                as_id_vec<InternalSurfaceId>(0, 1, 0, 0),
                as_id_vec<InternalSurfaceId>(0, 0, 1, 1),
                as_id_vec<InternalSurfaceId>(0, 0, 1, 1),
            },
        },
        // Geometric Surface 1
        // A | C | B
        //   4   5
        {
            as_id_vec<OptMatId>(2),
            as_id_vec<PhysSurfaceId>(4, 5),
            {
                as_id_vec<SurfaceModelId>(2, 1),
                as_id_vec<SurfaceModelId>(1, 0),
                as_id_vec<SurfaceModelId>(0, 1),
            },
            {
                as_id_vec<InternalSurfaceId>(1, 1),
                as_id_vec<InternalSurfaceId>(2, 2),
                as_id_vec<InternalSurfaceId>(2, 2),
            },
        },
        // Geometric Surface 2
        // A | B
        //   6
        {
            as_id_vec<OptMatId>(),
            as_id_vec<PhysSurfaceId>(6),
            {
                as_id_vec<SurfaceModelId>(0),
                as_id_vec<SurfaceModelId>(1),
                as_id_vec<SurfaceModelId>(0),
            },
            {
                as_id_vec<InternalSurfaceId>(2),
                as_id_vec<InternalSurfaceId>(3),
                as_id_vec<InternalSurfaceId>(3),
            },
        },
        // Geometric Surface 3 - default surface
        {
            std::vector<OptMatId>{},
            as_id_vec<PhysSurfaceId>(7),
            {
                as_id_vec<SurfaceModelId>(0),
                as_id_vec<SurfaceModelId>(1),
                as_id_vec<SurfaceModelId>(0),
            },
            {
                as_id_vec<InternalSurfaceId>(3),
                as_id_vec<InternalSurfaceId>(4),
                as_id_vec<InternalSurfaceId>(4),
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
        for (auto step : range(SurfacePhysicsOrder::size_))
        {
            EXPECT_VEC_EQ(expected_record.actions[step],
                          actual_record.actions[step]);
            EXPECT_VEC_EQ(expected_record.per_model_ids[step],
                          actual_record.per_model_ids[step]);
        }
    }

    // Check surface model data

    SurfaceOrderArray<std::vector<std::string_view>> expected_model_names;
    expected_model_names[SurfacePhysicsOrder::roughness] = {
        "polished",
        "smear",
        "gaussian",
    };
    expected_model_names[SurfacePhysicsOrder::reflectivity] = {
        "grid",
        "fresnel",
    };
    expected_model_names[SurfacePhysicsOrder::interaction] = {
        "dielectric-dielectric",
        "dielectric-metal",
    };

    for (auto step : range(SurfacePhysicsOrder::size_))
    {
        std::vector<std::string_view> model_names;
        for (auto const& model : params->models(step))
        {
            model_names.push_back(model->label());
        }
        EXPECT_VEC_EQ(expected_model_names[step], model_names);
    }
}

//---------------------------------------------------------------------------//
// Check initialization of surface physics views
TEST_F(SurfacePhysicsTest, init_surface_physics_view)
{
    auto expected_surfaces = as_id_vec<SurfaceId>(0, 1, 2, 2, 0, 1, 0);
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
                                              expected_orientations[track],
                                              OptMatId{0},
                                              OptMatId{1}};
    }

    // Check initialization
    std::vector<SurfaceId> surfaces;
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
        s_physics.subsurface_position(
            SurfaceTrackPosition(s_physics.num_positions() - 1));

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
            s_physics.subsurface_position(pos);

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
        s_physics = SurfacePhysicsView::Initializer{
            SurfaceId{2}, forward, OptMatId{0}, OptMatId{1}};

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        auto result = trace_directions(s_physics, directions);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_FALSE(s_physics.in_pre_volume());
        EXPECT_TRUE(s_physics.in_post_volume());

        TraceResult expected{as_id_vec<SurfaceTrackPosition>(0, 1),
                             as_id_vec<OptMatId>(0, 1),
                             as_id_vec<PhysSurfaceId>(6)};

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
        s_physics = SurfacePhysicsView::Initializer{
            SurfaceId{2}, reverse, OptMatId{1}, OptMatId{0}};

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_TRUE(s_physics.in_pre_volume());
        EXPECT_FALSE(s_physics.in_post_volume());

        auto result = trace_directions(s_physics, directions);

        EXPECT_TRUE(s_physics.is_crossing_boundary());
        EXPECT_FALSE(s_physics.in_pre_volume());
        EXPECT_TRUE(s_physics.in_post_volume());

        TraceResult expected{as_id_vec<SurfaceTrackPosition>(0, 1),
                             as_id_vec<OptMatId>(1, 0),
                             as_id_vec<PhysSurfaceId>(6)};

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
            reverse,
        };

        auto s_physics = this->surface_physics_view(TrackSlotId{2});
        s_physics = SurfacePhysicsView::Initializer{
            SurfaceId{0}, forward, OptMatId{0}, OptMatId{1}};

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
            as_id_vec<PhysSurfaceId>(0, 1, 1, 1, 2, 3, 3, 2, 1, 0)};

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
        s_physics = SurfacePhysicsView::Initializer{
            SurfaceId{1}, reverse, OptMatId{1}, OptMatId{0}};

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
            as_id_vec<PhysSurfaceId>(5, 4, 4, 5, 5, 4)};

        EXPECT_VEC_EQ(expected.position, result.position);
        EXPECT_VEC_EQ(expected.material, result.material);
        EXPECT_VEC_EQ(expected.interface, result.interface);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
