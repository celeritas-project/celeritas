//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/VolumeSurfaceSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/VolumeSurfaceSelector.hh"

#include "geocel/SurfaceParams.hh"
#include "geocel/SurfaceTestBase.hh"
#include "geocel/VolumeParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{

using OSurface = VolumeSurfaceSelector::OrientedSurface;
constexpr auto forward = SubsurfaceDirection::forward;
constexpr auto reverse = SubsurfaceDirection::reverse;

bool operator==(OSurface const& a, OSurface const& b)
{
    if (!a.surface && !b.surface)
    {
        return true;
    }

    return a.surface == b.surface && a.orientation == b.orientation;
}

std::ostream& operator<<(std::ostream& out, OSurface const& s)
{
    return out << "{" << s.surface.unchecked_get() << ", "
               << static_cast<int>(s.orientation) << "}";
}

namespace test
{

using namespace ::celeritas::test;

//---------------------------------------------------------------------------//
// MANY-SURFACES
//---------------------------------------------------------------------------//

using VolumeSurfaceSelectorTest = ::celeritas::test::ManySurfacesTestBase;

// Test surface selection for various pre and post volume instances
TEST_F(VolumeSurfaceSelectorTest, select_surface)
{
    auto const& surfaces = this->surfaces();
    auto const& volumes_ = this->volumes();

    auto select_surfaces = [&](VolumeInstanceId pre_vol_inst) {
        std::vector<OSurface> results;

        VolumeSurfaceSelector select{
            VolumeSurfaceView{surfaces.host_ref(),
                              volumes_.volume(pre_vol_inst)},
            pre_vol_inst};
        for (auto post_vol_inst :
             range(VolumeInstanceId{volumes_.num_volume_instances()}))
        {
            auto post_vol = volumes_.volume(post_vol_inst);
            if (!post_vol)
            {
                // Not used in geometry; this represents Geant4 skipping some
                // volumes in its list
                continue;
            }

            results.push_back(
                select(VolumeSurfaceView{surfaces.host_ref(), post_vol},
                       post_vol_inst));
        }

        return results;
    };

    {
        static OSurface const expected_surfaces[] = {
            {SurfaceId{2}, forward},
            {SurfaceId{5}, forward},
            {SurfaceId{6}, forward},
            {SurfaceId{2}, forward},
            {SurfaceId{2}, forward},
            {SurfaceId{2}, forward},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{0}));
    }
    {
        static OSurface const expected_surfaces[] = {
            {SurfaceId{2}, reverse},
            {SurfaceId{}, forward},
            {SurfaceId{3}, forward},
            {SurfaceId{}, reverse},
            {SurfaceId{}, reverse},
            {SurfaceId{}, reverse},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{1}));
    }
    {
        static OSurface const expected_surfaces[] = {
            {SurfaceId{0}, forward},
            {SurfaceId{}, reverse},
            {SurfaceId{1}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{2}));
    }
    {
        static OSurface const expected_surfaces[] = {
            {SurfaceId{2}, reverse},
            {SurfaceId{4}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{3}));
    }
    {
        static OSurface const expected_surfaces[] = {
            {SurfaceId{2}, reverse},
            {SurfaceId{8}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{4}));
    }
    {
        static OSurface const expected_surfaces[] = {
            {SurfaceId{2}, reverse},
            {SurfaceId{7}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
            {SurfaceId{}, forward},
        };

        EXPECT_VEC_EQ(expected_surfaces, select_surfaces(VolumeInstanceId{6}));
    }
}

//---------------------------------------------------------------------------//
// Explicitly check current precedence for mother-daughter boundaries
TEST_F(VolumeSurfaceSelectorTest, mother_daughter)
{
    // TODO: fix after Hayden's merge
    auto const& surfaces = this->surfaces();
    auto const& volumes_ = this->volumes();

    // Mother volume B
    VolumeInstanceId mother{0};
    VolumeSurfaceSelector select{
        VolumeSurfaceView{surfaces.host_ref(), volumes_.volume(mother)},
        mother};

    // Daughter volume C3
    VolumeInstanceId daughter{3};

    // Check precedence of selecting boundary surfaces for mother-daughter
    // relations Geant4: select daughter's boundary if present (SurfaceId{8})
    // Celeritas: select pre-volume first (SurfaceId{2})
    OSurface result{SurfaceId{2}, forward};
    EXPECT_EQ(result,
              select(VolumeSurfaceView{surfaces.host_ref(),
                                       volumes_.volume(daughter)},
                     daughter));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
