//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/VolumeSurfaceSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/VolumeSurfaceSelector.hh"

#include "geocel/SurfaceParams.hh"
#include "geocel/SurfaceTestBase.hh"

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

class VolumeSurfaceSelectorTest : public SurfaceTestBase
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test surface selection for various pre and post volume instances
TEST_F(VolumeSurfaceSelectorTest, select_surface)
{
    SurfaceParams surfaces{this->make_many_surfaces_inp(), volumes_};

    auto select_surfaces = [&](VolumeInstanceId pre_vol_inst) {
        std::vector<SurfaceId> results;

        VolumeSurfaceSelector select{
            surfaces.host_ref(), volumes_.volume(pre_vol_inst), pre_vol_inst};
        for (auto post_vol_inst :
             range(VolumeInstanceId{volumes_.num_volume_instances()}))
        {
            results.push_back(
                select(volumes_.volume(post_vol_inst), post_vol_inst));
        }

        return results;
    };

    {
        static SurfaceId const expected_surfaces[] = {
            SurfaceId{2},
            SurfaceId{5},
            SurfaceId{6},
            SurfaceId{2},
            SurfaceId{2},
            SurfaceId{2},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{0}));
    }
    {
        static SurfaceId const expected_surfaces[] = {
            SurfaceId{2},
            SurfaceId{},
            SurfaceId{3},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{1}));
    }
    {
        static SurfaceId const expected_surfaces[] = {
            SurfaceId{0},
            SurfaceId{},
            SurfaceId{1},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{2}));
    }
    {
        static SurfaceId const expected_surfaces[] = {
            SurfaceId{2},
            SurfaceId{4},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{3}));
    }
    {
        static SurfaceId const expected_surfaces[] = {
            SurfaceId{2},
            SurfaceId{8},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{4}));
    }
    {
        static SurfaceId const expected_surfaces[] = {
            SurfaceId{2},
            SurfaceId{7},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
            SurfaceId{},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{5}));
    }
}

//---------------------------------------------------------------------------//
// Explicitly check current precedence for mother-daughter boundaries
TEST_F(VolumeSurfaceSelectorTest, mother_daughter)
{
    SurfaceParams surfaces{
        ([&]() {
            auto surface_input = this->make_many_surfaces_inp();
            // Add a boundary surface to C volumes
            surface_input.surfaces.push_back(make_surface("c", VolumeId{2}));
            return surface_input;
        })(),
        volumes_};

    // Mother volume B
    VolumeInstanceId mother{0};
    VolumeSurfaceSelector select{
        surfaces.host_ref(), volumes_.volume(mother), mother};

    // Daughter volume C3
    VolumeInstanceId daughter{3};

    // Check precedence of selecting boundary surfaces for mother-daughter
    // relations Geant4: select daughter's boundary if present (SurfaceId{8})
    // Celeritas: select pre-volume first (SurfaceId{2})
    EXPECT_EQ(SurfaceId{2}, select(volumes_.volume(daughter), daughter));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
