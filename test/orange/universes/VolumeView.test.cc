//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VolumeView.test.cc
//---------------------------------------------------------------------------//
#include "orange/universes/VolumeView.hh"

#include "celeritas_config.h"
#include "celeritas_test.hh"
#include "base/Range.hh"
#include "../OrangeGeoTestBase.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class VolumeViewTest : public celeritas_test::OrangeGeoTestBase
{
  protected:
    using VolumeDataRef = VolumeView::VolumeDataRef;

    const VolumeDataRef& volume_ref() const
    {
        return this->params().host_ref().volumes;
    }

    void test_face_accessors(const VolumeView& volumes)
    {
        auto faces = volumes.faces();
        ASSERT_EQ(faces.size(), volumes.num_faces());

        for (auto face_id : range(FaceId{volumes.num_faces()}))
        {
            SurfaceId surf_id = faces[face_id.get()];
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
    ASSERT_EQ(1, this->volume_ref().size());

    VolumeView vol(this->volume_ref(), VolumeId{0});
    EXPECT_EQ(0, vol.num_faces());
    this->test_face_accessors(vol);

    // Test that nonexistent face returns "false" id
    EXPECT_EQ(FaceId{}, vol.find_face(SurfaceId{123}));
    if (CELERITAS_DEBUG)
    {
        // Disallow empty surfaces (burden is on the caller to check for null)
        EXPECT_THROW(vol.get_surface(FaceId{}), DebugError);
        EXPECT_THROW(vol.find_face(SurfaceId{}), DebugError);
    }
}

TEST_F(VolumeViewTest, five_volumes)
{
    if (!CELERITAS_USE_JSON)
    {
        GTEST_SKIP() << "JSON is not enabled";
    }

    this->build_geometry("five-volumes.org.json");

    std::vector<size_type> num_faces;

    for (auto vol_id : range(VolumeId{this->volume_ref().size()}))
    {
        VolumeView vol(this->volume_ref(), vol_id);
        num_faces.push_back(vol.num_faces());
        this->test_face_accessors(vol);
    }

    const unsigned long expected_num_faces[] = {1ul, 7ul, 7ul, 2ul, 11ul, 1ul};
    EXPECT_VEC_EQ(expected_num_faces, num_faces);
}
