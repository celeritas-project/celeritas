//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SenseCalculator.test.cc
//---------------------------------------------------------------------------//
#include "orange/universes/detail/SenseCalculator.hh"

#include <vector>
#include "base/Span.hh"
#include "orange/surfaces/Surfaces.hh"
#include "celeritas_test.hh"
#include "../../OrangeGeoTestBase.hh"

using celeritas::detail::SenseCalculator;
using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SenseCalculatorTest : public celeritas_test::OrangeGeoTestBase
{
  protected:
    using SurfaceDataRef = Surfaces::SurfaceDataRef;
    using VolumeDataRef  = VolumeView::VolumeDataRef;

    const SurfaceDataRef& surface_ref() const
    {
        return this->params_host_ref().surfaces;
    }
    const VolumeDataRef& volume_ref() const
    {
        return this->params_host_ref().volumes;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SenseCalculatorTest, one_volumes)
{
    {
        OneVolInput geo_inp;
        this->build_geometry(geo_inp);
    }

    // Test this degenerate case (no surfaces)
    SenseCalculator calc_senses(Surfaces{this->surface_ref()},
                                Real3{123, 345, 567},
                                this->sense_storage());

    auto result = calc_senses(VolumeView(this->volume_ref(), VolumeId{0}));
    EXPECT_EQ(0, result.senses.size());
    EXPECT_EQ(FaceId{}, result.face);
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
    VolumeView inner(this->volume_ref(), VolumeId{0});
    VolumeView outer(this->volume_ref(), VolumeId{1});

    {
        // Point is in the inner sphere
        SenseCalculator calc_senses(Surfaces{this->surface_ref()},
                                    Real3{0, 0.5, 0},
                                    this->sense_storage());
        {
            // Test inner sphere, not on a face
            auto result = calc_senses(inner);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::inside, result.senses[0]);
            EXPECT_EQ(FaceId{}, result.face);
        }
        {
            // Test not-sphere, not on a face
            auto result = calc_senses(outer);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::inside, result.senses[0]);
            EXPECT_EQ(FaceId{}, result.face);
        }
    }
    {
        // Point is in on the boundary: should register as "on" the face
        SenseCalculator calc_senses(Surfaces{this->surface_ref()},
                                    Real3{1.5, 0, 0},
                                    this->sense_storage());
        {
            auto result = calc_senses(inner);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::outside, result.senses[0]);
            EXPECT_EQ(FaceId{0}, result.face);
        }
    }
    {
        // Point is in the outer sphere
        SenseCalculator calc_senses(Surfaces{this->surface_ref()},
                                    Real3{2, 0, 0},
                                    this->sense_storage());
        {
            auto result = calc_senses(inner);
            ASSERT_EQ(1, result.senses.size());
            EXPECT_EQ(Sense::outside, result.senses[0]);
            EXPECT_EQ(FaceId{}, result.face);
        }
    }
}

TEST_F(SenseCalculatorTest, five_volumes)
{
    if (!CELERITAS_USE_JSON)
    {
        GTEST_SKIP() << "JSON is not enabled";
    }

    this->build_geometry("five-volumes.org.json");
    // this->describe(std::cout);

    // Volume definitions
    VolumeView vol_b(this->volume_ref(), VolumeId{2});
    VolumeView vol_c(this->volume_ref(), VolumeId{3});
    VolumeView vol_e(this->volume_ref(), VolumeId{5});

    {
        // Point is in the inner sphere
        SenseCalculator calc_senses(Surfaces{this->surface_ref()},
                                    Real3{-.25, -.25, 0},
                                    this->sense_storage());
        {
            // Test inner sphere
            auto result = calc_senses(vol_e);
            EXPECT_EQ("{-}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{}, result.face);
        }
        {
            // Test between spheres
            auto result = calc_senses(vol_c);
            EXPECT_EQ("{- -}", senses_to_string(result.senses));
        }
        {
            // Test square (faces: 1 through 7)
            auto result = calc_senses(vol_b);
            EXPECT_EQ("{- + - - - - +}", senses_to_string(result.senses));
        }
    }
    {
        // Point is between spheres, on square edge
        SenseCalculator calc_senses(Surfaces{this->surface_ref()},
                                    Real3{0.5, -0.25, 0},
                                    this->sense_storage());
        {
            // Test inner sphere
            auto result = calc_senses(vol_e);
            EXPECT_EQ("{+}", senses_to_string(result.senses));
            EXPECT_EQ(FaceId{}, result.face);
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
            EXPECT_EQ(FaceId{4}, result.face);
        }
    }
}
