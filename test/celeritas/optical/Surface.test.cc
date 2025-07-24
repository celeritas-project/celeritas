//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Surface.test.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <set>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionStateStore.hh"
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

constexpr SubsurfaceDirection forward = SubsurfaceDirection::forward;
constexpr SubsurfaceDirection reverse = SubsurfaceDirection::reverse;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceTest : public OpticalMockTestBase
{
  protected:
    Array<size_type, 3> const expected_num_subsurfaces{
        3,
        5,
        2,
    };
    Array<SurfaceId, 5> const expected_track_surfaces{
        SurfaceId{0},
        SurfaceId{1},
        SurfaceId{1},
        SurfaceId{2},
        SurfaceId{2},
    };
    Array<SubsurfaceDirection, 5> const expected_surface_orientations{
        forward,
        forward,
        reverse,
        reverse,
        reverse,
    };

    void SetUp() override {}

    SPConstSurfacePhysics build_surface_physics() override
    {
        SurfacePhysicsParams::Input input;

        input.action_registry = this->action_reg().get();

        // Make some mock surface records
        for (size_type n : expected_num_subsurfaces)
        {
            input.num_subsurface_interfaces.push_back(n);
        }

        return std::make_shared<SurfacePhysicsParams const>(std::move(input));
    }

    void initialize_states()
    {
        size_type num_tracks = expected_track_surfaces.size();
        surface_physics_state_
            = CollectionStateStore<SurfacePhysicsStateData, MemSpace::host>(
                num_tracks);
        CELER_ASSERT(surface_physics_state_.ref().size() == num_tracks);

        for (auto i : range(TrackSlotId{5}))
        {
            this->make_surface_view(i) = SurfacePhysicsView::Initializer{
                expected_track_surfaces[i.get()],
                expected_surface_orientations[i.get()]};
        }
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
// Test initialization
TEST_F(SurfaceTest, init_params)
{
    auto params = this->surface_physics();

    EXPECT_EQ(ActionId{0}, params->init_boundary_action());
    EXPECT_EQ(ActionId{1}, params->post_boundary_action());

    auto const& data = params->host_ref();

    EXPECT_TRUE(data);
    ASSERT_EQ(expected_num_subsurfaces.size(), data.surfaces.size());

    for (auto i : range(expected_num_subsurfaces.size()))
    {
        auto const& surface = data.surfaces[SurfaceId(i)];
        EXPECT_TRUE(surface);
        EXPECT_EQ(expected_num_subsurfaces[i],
                  surface.subsurface_interfaces.size());
        EXPECT_EQ(expected_num_subsurfaces[i] + 1,
                  surface.subsurface_materials.size());
    }
}

//---------------------------------------------------------------------------//
// Test surface physics view initialization
TEST_F(SurfaceTest, surface_physics_view_init)
{
    auto params = this->surface_physics();
    this->initialize_states();

    // Views initialized in initialize_states, check here

    std::vector<SurfaceId> surfaces;

    for (auto i : range(TrackSlotId{5}))
    {
        auto surface = this->make_surface_view(i);

        surfaces.push_back(surface.surface_id());

        EXPECT_EQ(SubsurfaceMaterialId{0}, surface.subsurface_material());
        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_EQ(
            expected_num_subsurfaces[expected_track_surfaces[i.get()].get()],
            surface.num_subsurface_interfaces());
    }

    EXPECT_VEC_EQ(expected_track_surfaces, surfaces);
}

//---------------------------------------------------------------------------//
// Test subsurface interface crossing
TEST_F(SurfaceTest, cross_subsurface)
{
    auto params = this->surface_physics();
    this->initialize_states();

    auto trace_directions
        = [&](SurfacePhysicsView& surface,
              std::vector<SubsurfaceDirection> const& directions) {
              std::vector<size_type> subsurfaces;
              std::vector<size_type> subsurface_records;
              std::vector<size_type> subinterface_records;

              subsurfaces.push_back(surface.subsurface_material().get());
              subsurface_records.push_back(
                  surface.subsurface_material_record().get());

              for (auto d : directions)
              {
                  subinterface_records.push_back(
                      surface.subsurface_interface_record(d).get());

                  surface.cross_subsurface_interface(d);

                  subsurfaces.push_back(surface.subsurface_material().get());
                  subsurface_records.push_back(
                      surface.subsurface_material_record().get());
              }

              return std::make_tuple(std::move(subsurfaces),
                                     std::move(subsurface_records),
                                     std::move(subinterface_records));
          };

    {
        auto surface = this->make_surface_view(TrackSlotId{0});

        // Directly forward
        std::vector<SubsurfaceDirection> const directions{
            forward,
            forward,
            forward,
        };
        static size_type const expected_subsurfaces[] = {
            0,
            1,
            2,
            3,
        };
        static size_type const expected_subsurface_records[] = {
            0,
            1,
            2,
            3,
        };
        static size_type const expected_subinterface_records[] = {
            0,
            1,
            2,
        };

        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());

        auto [subsurfaces, subsurface_records, subinterface_records]
            = trace_directions(surface, directions);

        EXPECT_FALSE(surface.in_pre_volume());
        EXPECT_TRUE(surface.in_post_volume());

        EXPECT_VEC_EQ(expected_subsurfaces, subsurfaces);
        EXPECT_VEC_EQ(expected_subsurface_records, subsurface_records);
        EXPECT_VEC_EQ(expected_subinterface_records, subinterface_records);
    }
    {
        auto surface = this->make_surface_view(TrackSlotId{1});

        // Out and back
        std::vector<SubsurfaceDirection> const directions{
            forward,
            forward,
            forward,
            reverse,
            reverse,
            reverse,
        };
        static size_type const expected_subsurfaces[] = {
            0,
            1,
            2,
            3,
            2,
            1,
            0,
        };
        static size_type const expected_subsurface_records[] = {
            0,
            1,
            2,
            3,
            2,
            1,
            0,
        };
        static size_type const expected_subinterface_records[] = {
            0,
            1,
            2,
            2,
            1,
            0,
        };

        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());

        auto [subsurfaces, subsurface_records, subinterface_records]
            = trace_directions(surface, directions);

        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());

        EXPECT_VEC_EQ(expected_subsurfaces, subsurfaces);
        EXPECT_VEC_EQ(expected_subsurface_records, subsurface_records);
        EXPECT_VEC_EQ(expected_subinterface_records, subinterface_records);
    }
    {
        auto surface = this->make_surface_view(TrackSlotId{2});

        // (Reversed) out
        std::vector<SubsurfaceDirection> const directions{
            forward, forward, forward, forward, forward};
        static size_type const expected_subsurfaces[] = {
            0,
            1,
            2,
            3,
            4,
            5,
        };
        static size_type const expected_subsurface_records[] = {
            5,
            4,
            3,
            2,
            1,
            0,
        };
        static size_type const expected_subinterface_records[] = {
            4,
            3,
            2,
            1,
            0,
        };

        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());

        auto [subsurfaces, subsurface_records, subinterface_records]
            = trace_directions(surface, directions);

        EXPECT_FALSE(surface.in_pre_volume());
        EXPECT_TRUE(surface.in_post_volume());

        EXPECT_VEC_EQ(expected_subsurfaces, subsurfaces);
        EXPECT_VEC_EQ(expected_subsurface_records, subsurface_records);
        EXPECT_VEC_EQ(expected_subinterface_records, subinterface_records);
    }
    {
        auto surface = this->make_surface_view(TrackSlotId{3});

        // (Reversed) out and back
        std::vector<SubsurfaceDirection> const directions{
            forward,
            forward,
            reverse,
            reverse,
        };
        static size_type const expected_subsurfaces[] = {
            0,
            1,
            2,
            1,
            0,
        };
        static size_type const expected_subsurface_records[] = {
            2,
            1,
            0,
            1,
            2,
        };
        static size_type const expected_subinterface_records[] = {
            1,
            0,
            0,
            1,
        };

        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());

        auto [subsurfaces, subsurface_records, subinterface_records]
            = trace_directions(surface, directions);

        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());

        EXPECT_VEC_EQ(expected_subsurfaces, subsurfaces);
        EXPECT_VEC_EQ(expected_subsurface_records, subsurface_records);
        EXPECT_VEC_EQ(expected_subinterface_records, subinterface_records);
    }
}

//---------------------------------------------------------------------------//
// Test catching debug errors crossing off of the surface
TEST_F(SurfaceTest, TEST_IF_CELERITAS_DEBUG(crossing_errors))
{
    auto params = this->surface_physics();
    this->initialize_states();

    {
        auto surface = this->make_surface_view(TrackSlotId{3});

        // Move past pre-volume
        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());
        EXPECT_EQ(SubsurfaceMaterialId{0}, surface.subsurface_material());

        EXPECT_THROW(surface.cross_subsurface_interface(reverse), DebugError);
    }
    {
        auto surface = this->make_surface_view(TrackSlotId{4});

        // Move past post-volume
        EXPECT_TRUE(surface.in_pre_volume());
        EXPECT_FALSE(surface.in_post_volume());
        EXPECT_EQ(SubsurfaceMaterialId{0}, surface.subsurface_material());

        surface.cross_subsurface_interface(forward);
        surface.cross_subsurface_interface(forward);

        EXPECT_THROW(surface.cross_subsurface_interface(forward), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
