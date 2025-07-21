//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeant.test.cc
//---------------------------------------------------------------------------//
#include <string>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/Types.hh"
#include "geocel/GenericGeoParameterizedTest.hh"
#include "geocel/GeoTests.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/rasterize/SafetyImager.hh"

#include "OrangeGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GeantOrangeTest : public OrangeGeoTestBase
{
  protected:
    void SetUp() final { this->geometry(); }

    SPConstGeo build_geometry() final
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::error};

        auto filename = this->geometry_basename() + std::string{".gdml"};
        auto result = Params::from_gdml(test_data_path("geocel", filename));

        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
        return result;
    }

    Constant unit_length() const final { return lengthunits::centimeter; }
};

//---------------------------------------------------------------------------//
using MultiLevelTest
    = GenericGeoParameterizedTest<GeantOrangeTest, MultiLevelGeoTest>;

TEST_F(MultiLevelTest, DISABLED_model)
{
    this->impl().test_model();
}

TEST_F(MultiLevelTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

class PincellTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "pincell"; }
};

TEST_F(PincellTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-12, -12, 0});
    inp.upper_right = from_cm({12, 12, 0});
    inp.rightward = {1.0, 0.0, 0.0};
    inp.vertical_pixels = 16;

    write_image(ImageParams{inp}, "org-pincell-xy-mid.jsonl");

    inp.lower_left[2] = inp.upper_right[2] = from_cm(-5.5);
    write_image(ImageParams{inp}, "org-pincell-xy-lo.jsonl");

    inp.lower_left = from_cm({-12, 0, -12});
    inp.upper_right = from_cm({12, 0, 12});
    write_image(ImageParams{inp}, "org-pincell-xz-mid.jsonl");
}

//---------------------------------------------------------------------------//
using PolyhedraTest
    = GenericGeoParameterizedTest<GeantOrangeTest, PolyhedraGeoTest>;

TEST_F(PolyhedraTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using ReplicaTest
    = GenericGeoParameterizedTest<GeantOrangeTest, ReplicaGeoTest>;

TEST_F(ReplicaTest, trace)
{
    this->impl().test_trace();
}

TEST_F(ReplicaTest, DISABLED_volume_stack)
{
    this->impl().test_volume_stack();
}

//---------------------------------------------------------------------------//
using SimpleCmsTest
    = GenericGeoParameterizedTest<GeantOrangeTest, SimpleCmsGeoTest>;

TEST_F(SimpleCmsTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using TestEm3Test
    = GenericGeoParameterizedTest<GeantOrangeTest, TestEm3GeoTest>;

TEST_F(TestEm3Test, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using TestEm3FlatTest
    = GenericGeoParameterizedTest<GeantOrangeTest, TestEm3FlatGeoTest>;

TEST_F(TestEm3FlatTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
class TilecalPlugTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "tilecal-plug"; }
};

TEST_F(TilecalPlugTest, trace)
{
    {
        SCOPED_TRACE("lo x");
        auto result = this->track({5.75, 0.01, -40}, {0, 0, 1});
        static char const* const expected_volumes[] = {
            "Tile_ITCModule",
            "Tile_Plug1Module",
            "Tile_Absorber",
            "Tile_Plug1Module",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {22.9425, 0.115, 42, 37};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("hi x");
        auto result = this->track({6.25, 0.01, -40}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"Tile_ITCModule", "Tile_Absorber", "Tile_Plug1Module"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {23.0575, 42, 37};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//
using TransformedBoxTest
    = GenericGeoParameterizedTest<GeantOrangeTest, TransformedBoxGeoTest>;

TEST_F(TransformedBoxTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TransformedBoxTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

class TwoBoxesTest
    : public GenericGeoParameterizedTest<GeantOrangeTest, TwoBoxesGeoTest>
{
};

TEST_F(TwoBoxesTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TwoBoxesTest, track)
{
    // Templated test
    TwoBoxesGeoTest::test_detailed_tracking(this);
}

//---------------------------------------------------------------------------//
using ZnenvTest = GenericGeoParameterizedTest<GeantOrangeTest, ZnenvGeoTest>;

TEST_F(ZnenvTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
