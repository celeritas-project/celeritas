//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/VolumeSurfaceSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/VolumeSurfaceSelector.hh"

#include <iostream>

#include "corecel/data/CollectionStateStore.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/CoreGeoTestBase.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"

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

class VolumeSurfaceSelectorTest : public OnlyGeoTestBase,
                                  public GlobalGeoTestBase,
                                  public OnlyCoreTestBase
{
  public:
    std::string_view geometry_basename() const override
    {
        return "optical-surfaces";
    }

  protected:
    void SetUp() override
    {
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
        {
            auto model = this->geometry()->make_model_input();

            volumes_ = std::make_shared<VolumeParams>(model.volumes);
            CELER_ENSURE(volumes_);

            surfaces_
                = std::make_shared<SurfaceParams>(model.surfaces, *volumes_);
            CELER_ENSURE(surfaces_);
        }
    }

    // Select surface for all volume instances besides the pre volume instance
    std::vector<SurfaceId> select_surfaces(VolumeInstanceId pre_vol_inst) const
    {
        std::vector<SurfaceId> results;

        VolumeSurfaceSelector select{surfaces_->host_ref(),
                                     volumes_->volume(pre_vol_inst),
                                     pre_vol_inst};
        for (auto post_vol_inst :
             range(VolumeInstanceId{volumes_->num_volume_instances()}))
        {
            if (post_vol_inst != pre_vol_inst)
            {
                results.push_back(
                    select(volumes_->volume(post_vol_inst), post_vol_inst));
            }
        }

        return results;
    }

    std::shared_ptr<VolumeParams const> volumes_;
    std::shared_ptr<SurfaceParams const> surfaces_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test surface selection for various pre and post volume instances
TEST_F(VolumeSurfaceSelectorTest, select_surface)
{
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
    {
        {
            static SurfaceId const expected_surfaces[] = {
                SurfaceId{0},
                SurfaceId{0},
                SurfaceId{0},
                SurfaceId{0},
            };

            EXPECT_VEC_EQ(expected_surfaces,
                          this->select_surfaces(VolumeInstanceId{0}));
        }
        {
            static SurfaceId const expected_surfaces[] = {
                SurfaceId{1},
                SurfaceId{2},
                SurfaceId{1},
                SurfaceId{1},
            };

            EXPECT_VEC_EQ(expected_surfaces,
                          this->select_surfaces(VolumeInstanceId{1}));
        }
        {
            static SurfaceId const expected_surfaces[] = {
                SurfaceId{0},
                SurfaceId{3},
                SurfaceId{4},
                SurfaceId{},
            };

            EXPECT_VEC_EQ(expected_surfaces,
                          this->select_surfaces(VolumeInstanceId{2}));
        }
        {
            static SurfaceId const expected_surfaces[] = {
                SurfaceId{1},
                SurfaceId{1},
                SurfaceId{1},
                SurfaceId{1},
            };

            EXPECT_VEC_EQ(expected_surfaces,
                          this->select_surfaces(VolumeInstanceId{3}));
        }
        {
            static SurfaceId const expected_surfaces[] = {
                SurfaceId{0},
                SurfaceId{1},
                SurfaceId{},
                SurfaceId{1},
            };

            EXPECT_VEC_EQ(expected_surfaces,
                          this->select_surfaces(VolumeInstanceId{4}));
        }
    }
}

//---------------------------------------------------------------------------//
// Check selector correctly uses volume instances from a geo view
TEST_F(VolumeSurfaceSelectorTest, geo_view_wrapper)
{
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
    {
        CollectionStateStore<CoreGeoStateData, MemSpace::host> host_state{
            this->geometry()->host_ref(), 1};
        GeoTrackView geo{
            this->geometry()->host_ref(), host_state.ref(), TrackSlotId{0}};
        geo = GeoTrackInitializer{Real3{0, 0, 0}, Real3{1, 0, 0}};

        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{2}, geo.volume_instance_id());

        // move across a surface and return the selected surface ID
        auto cross_surface = [&]() {
            geo.find_next_step();
            geo.move_to_boundary();

            VolumeSurfaceSelector select{surfaces_->host_ref(), geo};

            geo.cross_boundary();

            return select(geo);
        };

        // tube1_mid -> world
        EXPECT_FALSE(cross_surface());
        EXPECT_EQ(VolumeId{3}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{4}, geo.volume_instance_id());

        // world -> lar_sphere
        EXPECT_EQ(SurfaceId{0}, cross_surface());
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{0}, geo.volume_instance_id());

        // tube2_below_pv -> tube1_mid_pv
        geo = GeoTrackInitializer{Real3{0, 0, -15}, Real3{0, 0, 1}};
        EXPECT_EQ(VolumeId{2}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{1}, geo.volume_instance_id());

        EXPECT_EQ(SurfaceId{2}, cross_surface());
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{2}, geo.volume_instance_id());

        // tube1_mid_pv -> tube2_above_pv
        EXPECT_EQ(SurfaceId{4}, cross_surface());
        EXPECT_EQ(VolumeId{2}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{3}, geo.volume_instance_id());

        // tube2_above_pv -> world
        EXPECT_EQ(SurfaceId{1}, cross_surface());
        EXPECT_EQ(VolumeId{3}, geo.volume_id());
        EXPECT_EQ(VolumeInstanceId{4}, geo.volume_instance_id());

        // world -> outside
        EXPECT_FALSE(cross_surface());
        EXPECT_TRUE(geo.is_outside());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
