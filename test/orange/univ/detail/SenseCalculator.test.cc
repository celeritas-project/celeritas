//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SenseCalculator.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/SenseCalculator.hh"

#include "orange/OrangeGeoTestBase.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"
#include "orange/univ/VolumeView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
// DETAIL TESTS
//---------------------------------------------------------------------------//

TEST(Types, OnFace)
{
    // Null face
    OnFace not_face;
    EXPECT_FALSE(not_face);
    EXPECT_FALSE(not_face.id());
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(not_face.sense(), DebugError);
    }
    EXPECT_NO_THROW(not_face.unchecked_sense());

    // On a face
    OnFace face{FaceId{3}, Sense::outside};
    EXPECT_TRUE(face);
    EXPECT_EQ(FaceId{3}, face.id());
    EXPECT_EQ(Sense::outside, face.sense());
    EXPECT_EQ(Sense::outside, face.unchecked_sense());
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SenseCalculatorTest : public ::celeritas::test::OrangeGeoTestBase
{
  protected:
    VolumeView make_volume_view(LocalVolumeId v) const
    {
        CELER_EXPECT(v);
        auto const& host_ref = this->host_params();
        return VolumeView{host_ref, host_ref.simple_units[SimpleUnitId{0}], v};
    }

    LocalSurfaceVisitor make_surf_visitor() const
    {
        return LocalSurfaceVisitor(this->host_params(), SimpleUnitId{0});
    }

    //! Access the shared CPU storage space for senses
    Span<SenseValue> sense_storage()
    {
        return this->host_state().temp_sense[AllItems<SenseValue>{}];
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SenseCalculatorTest, one_volume)
{
    {
        OneVolInput geo_inp;
        this->build_geometry(geo_inp);
    }

    // Test this degenerate case (no surfaces)
    SenseCalculator calc_senses(
        this->make_surf_visitor(), Real3{123, 345, 567}, this->sense_storage());

    auto result = calc_senses(this->make_volume_view(LocalVolumeId{0}));
    EXPECT_EQ(0, result.senses.size());
    EXPECT_FALSE(result.face);
}

TEST_F(SenseCalculatorTest, two_volumes)
{
    {
        TwoVolInput geo_inp;
        geo_inp.radius = 1.5;
        this->build_geometry(geo_inp);
    }

    // Note that since these have the same faces, the results should be the
    // same for both.
    VolumeView outer = this->make_volume_view(LocalVolumeId{0});
    VolumeView inner = this->make_volume_view(LocalVolumeId{1});

    {
        // Point is in the inner sphere
        SenseCalculator calc_senses(
            this->make_surf_visitor(), Real3{0, 0.5, 0}, this->sense_storage());
        {
            // Test inner sphere, not on a face
            auto result = calc_senses(inner);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::inside, result.senses[0]);
            EXPECT_FALSE(result.face);
        }
        {
            // Test not-sphere, not on a face
            auto result = calc_senses(outer);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::inside, result.senses[0]);
            EXPECT_FALSE(result.face);
        }
    }
    {
        // Point is in on the boundary: should register as "on" the face
        SenseCalculator calc_senses(
            this->make_surf_visitor(), Real3{1.5, 0, 0}, this->sense_storage());
        {
            auto result = calc_senses(inner);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::outside, result.senses[0]);
            EXPECT_EQ(FaceId{0}, result.face.id());
            EXPECT_EQ(Sense::outside, result.face.sense());
        }
        {
            auto result = calc_senses(inner, OnFace{FaceId{0}, Sense::inside});
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::inside, result.senses[0]);
            EXPECT_EQ(FaceId{0}, result.face.id());
            EXPECT_EQ(Sense::inside, result.face.sense());
        }
    }
    {
        // Point is in the outer sphere
        SenseCalculator calc_senses(
            this->make_surf_visitor(), Real3{2, 0, 0}, this->sense_storage());
        {
            auto result = calc_senses(inner);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::outside, result.senses[0]);
            EXPECT_FALSE(result.face);
        }
    }
}

TEST_F(SenseCalculatorTest, five_volumes)
{
    this->build_geometry("five-volumes.org.json");
    // this->describe(std::cout);

    // Volume definitions
    VolumeView vol_b = this->make_volume_view(LocalVolumeId{2});
    VolumeView vol_c = this->make_volume_view(LocalVolumeId{3});
    VolumeView vol_e = this->make_volume_view(LocalVolumeId{5});

    {
        // Point is in the inner sphere
        SenseCalculator calc_senses(this->make_surf_visitor(),
                                    Real3{-.25, -.25, 0},
                                    this->sense_storage());
        {
            // Test inner sphere
            auto result = calc_senses(vol_e);
            EXPECT_EQ("{-}", senses_to_string(result.senses));
            EXPECT_FALSE(result.face);
        }
        {
            // Test between spheres
            auto result = calc_senses(vol_c);
            EXPECT_EQ("{- -}", senses_to_string(result.senses));
        }
        {
            // Test square (faces: 3, 5, 6, 7, 8, 9, 10)
            auto result = calc_senses(vol_b);
            EXPECT_EQ("{- + - - - - +}", senses_to_string(result.senses));
        }
    }
    {
        // Point is between spheres, on square edge (surface 8)
        SenseCalculator calc_senses(this->make_surf_visitor(),
                                    Real3{0.5, -0.25, 0},
                                    this->sense_storage());
        {
            // Test inner sphere
            auto result = calc_senses(vol_e);
            EXPECT_EQ("{+}", senses_to_string(result.senses));
            EXPECT_FALSE(result.face);
        }
        {
            // Test between spheres
            auto result = calc_senses(vol_c);
            EXPECT_EQ("{- +}", senses_to_string(result.senses));
        }
        {
            // Test square (faces: 1 through 7)
            auto result = calc_senses(vol_b);
            EXPECT_EQ("{- + - - + - +}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{4}, result.face.id());
            EXPECT_EQ(Sense::outside, result.face.sense());
        }
        {
            // Test square with correct face (surface 8, face 4)
            auto result = calc_senses(vol_b, OnFace{FaceId{4}, Sense::outside});
            EXPECT_EQ("{- + - - + - +}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{4}, result.face.id());
            EXPECT_EQ(Sense::outside, result.face.sense());
        }
        {
            // Test square with flipped sense
            auto result = calc_senses(vol_b, OnFace{FaceId{4}, Sense::inside});
            EXPECT_EQ("{- + - - - - +}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{4}, result.face.id());
            EXPECT_EQ(Sense::inside, result.face.sense());
        }
        {
            // Test square with "incorrect" face that gets assigned anyway
            auto result = calc_senses(vol_b, OnFace{FaceId{1}, Sense::inside});
            EXPECT_EQ("{- - - - + - +}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{1}, result.face.id());
            EXPECT_EQ(Sense::inside, result.face.sense());
        }
        if (CELERITAS_DEBUG)
        {
            // Out-of-range face ID
            EXPECT_THROW(calc_senses(vol_b, OnFace{FaceId{8}, Sense::inside}),
                         DebugError);
        }
    }
    {
        // Point is exactly on the lower right corner of b. If a face isn't
        // given then the lower face ID will be the one considered "on".
        // +x = surface 9 = face 5
        // -y = surface 10 = face 6
        SenseCalculator calc_senses(this->make_surf_visitor(),
                                    Real3{1.5, -1.0, 0},
                                    this->sense_storage());
        {
            // Test natural sense
            auto result = calc_senses(vol_b);
            EXPECT_EQ("{- + - + + + +}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{5}, result.face.id());
            EXPECT_EQ(Sense::outside, result.face.sense());
        }
        {
            // Test with lower face, flipped sense
            auto result = calc_senses(vol_b, OnFace{FaceId{5}, Sense::inside});
            EXPECT_EQ("{- + - + + - +}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{5}, result.face.id());
            EXPECT_EQ(Sense::inside, result.face.sense());
        }
        {
            // Test with right face, flipped sense
            auto result = calc_senses(vol_b, OnFace{FaceId{6}, Sense::inside});
            EXPECT_EQ("{- + - + + + -}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{6}, result.face.id());
            EXPECT_EQ(Sense::inside, result.face.sense());
        }
    }
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
