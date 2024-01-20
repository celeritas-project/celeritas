//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/VolumeView.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/VolumeView.hh"

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "orange/OrangeGeoTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class VolumeViewTest : public OrangeGeoTestBase
{
  protected:
    VolumeView make_view(LocalVolumeId v) const
    {
        CELER_EXPECT(v);
        auto const& host_ref = this->host_params();
        return VolumeView{host_ref, host_ref.simple_units[SimpleUnitId{0}], v};
    }

    void test_face_accessors(VolumeView const& volumes)
    {
        auto faces = volumes.faces();
        ASSERT_EQ(faces.size(), volumes.num_faces());

        for (auto face_id : range(FaceId{volumes.num_faces()}))
        {
            LocalSurfaceId surf_id = faces[face_id.get()];
            EXPECT_EQ(surf_id, volumes.get_surface(face_id));
            EXPECT_EQ(face_id, volumes.find_face(surf_id));
        }
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(VolumeViewTest, one_volume)
{
    // Build a volume that's infinite
    this->build_geometry(OneVolInput{});
    ASSERT_EQ(1, this->num_volumes());

    VolumeView vol = this->make_view(LocalVolumeId{0});
    EXPECT_EQ(0, vol.num_faces());
    this->test_face_accessors(vol);

    // Test that nonexistent face returns "false" id
    EXPECT_EQ(FaceId{}, vol.find_face(LocalSurfaceId{123}));
    if (CELERITAS_DEBUG)
    {
        // Disallow empty surfaces (burden is on the caller to check for null)
        EXPECT_THROW(vol.get_surface(FaceId{}), DebugError);
        EXPECT_THROW(vol.find_face(LocalSurfaceId{}), DebugError);
    }
}

TEST_F(VolumeViewTest, TEST_IF_CELERITAS_JSON(five_volumes))
{
    this->build_geometry("five-volumes.org.json");

    std::vector<size_type> num_faces;
    std::vector<logic_int> flags;

    for (auto vol_id : range(LocalVolumeId{this->num_volumes()}))
    {
        VolumeView vol = this->make_view(vol_id);
        num_faces.push_back(vol.num_faces());
        this->test_face_accessors(vol);
    }

    size_type const expected_num_faces[] = {1u, 7u, 7u, 2u, 11u, 1u};
    EXPECT_VEC_EQ(expected_num_faces, num_faces);

    {
        VolumeView vol = this->make_view(LocalVolumeId{0});
        EXPECT_FALSE(vol.internal_surfaces());
        EXPECT_FALSE(vol.implicit_vol());
        EXPECT_TRUE(vol.simple_safety());
        EXPECT_TRUE(vol.simple_intersection());
    }
    {
        VolumeView vol = this->make_view(LocalVolumeId{4});
        EXPECT_TRUE(vol.internal_surfaces());
        EXPECT_FALSE(vol.implicit_vol());
        EXPECT_TRUE(vol.simple_safety());
        EXPECT_FALSE(vol.simple_intersection());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
