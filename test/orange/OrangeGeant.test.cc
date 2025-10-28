//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeant.test.cc
//---------------------------------------------------------------------------//
#include <string>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/StringSimplifier.hh"
#include "corecel/Types.hh"
#include "geocel/CheckedGeoTrackView.hh"
#include "geocel/GenericGeoParameterizedTest.hh"
#include "geocel/GeoTests.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/rasterize/SafetyImager.hh"
#include "orange/Debug.hh"

#include "OrangeTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GeantOrangeTest : public OrangeTestBase
{
  protected:
    void SetUp() final { this->geometry(); }

    //! Check log messages
    SPConstGeo build_geometry() const final
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::error};
        auto result = OrangeTestBase::build_geometry();
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
        return result;
    }

    Constant unit_length() const final { return lengthunits::centimeter; }
};

//---------------------------------------------------------------------------//
using FourLevelsTest
    = GenericGeoParameterizedTest<GeantOrangeTest, FourLevelsGeoTest>;

TEST_F(FourLevelsTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(FourLevelsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(FourLevelsTest, consecutive_compute)
{
    this->impl().test_consecutive_compute();
}

TEST_F(FourLevelsTest, detailed_track)
{
    this->impl().test_detailed_tracking();
}

//---------------------------------------------------------------------------//
using LarSphereTest
    = GenericGeoParameterizedTest<GeantOrangeTest, LarSphereGeoTest>;

TEST_F(LarSphereTest, trace)
{
    this->impl().test_trace();
}

TEST_F(LarSphereTest, DISABLED_volume_stack)
{
    this->impl().test_volume_stack();
}

//---------------------------------------------------------------------------//
class MultiLevelTest
    : public GenericGeoParameterizedTest<GeantOrangeTest, MultiLevelGeoTest>
{
};

TEST_F(MultiLevelTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

class PincellTest : public GeantOrangeTest
{
    std::string_view gdml_basename() const final { return "pincell"; }
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
class ReplicaTest
    : public GenericGeoParameterizedTest<GeantOrangeTest, ReplicaGeoTest>
{
  public:
    //! Distance is slightly off for single precision
    GenericGeoTrackingTolerance tracking_tol() const override
    {
        auto result = GeantOrangeTest::tracking_tol();

        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
        {
            result.distance *= 10;
        }

        return result;
    }
};

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
    std::string_view gdml_basename() const final { return "tilecal-plug"; }
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

/*!
 * Cross into a new volume and then reflect into the old.
 *
 * This is how optical physics is performed: we enter the new volume to
 * determine its characteristics, then apply the optical surface crossing,
 * which might reflect back into the original.
 */
TEST_F(TwoBoxesTest, reentrant)
{
    constexpr auto dx = real_type{1} / constants::sqrt_two;

    // Starting left of edge (-), headed down right (+,-)
    CheckedGeoTrackView geo{this->make_geo_track_view_interface()};
    geo = this->make_initializer({5 - dx, dx, 0}, {dx, -dx, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces up to a distance of 4 units away
    auto next = geo.find_next_step(from_cm(4.0));
    EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    // Move to boundary (-; +,-)
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("inner", this->volume_name(geo));

    // Cross into the new volume, needed for optical physics (+; +,-)
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("world", this->volume_name(geo));

    // Reflect normal to surface  (+; -,-)
    geo.set_dir(Real3{-dx, -dx, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("world", this->volume_name(geo));

    // Cross back into previous volume (-; -,-)
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("inner", this->volume_name(geo));

    // Find the next boundary and make sure that nearer distances aren't
    // accepted
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(10.0, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    EXPECT_TRUE(geo.is_on_boundary());
}

/*!
 * Instead of crossing into a new volume, reflect without exiting.
 *
 * This simulates a looping track almost tangent to a geometry boundary.
 * The end-of-step direction is changed to account for the momentum vector's
 * end-of-step state, and the boundary isn't actually exited when we call cross
 * boundary.
 */
TEST_F(TwoBoxesTest, tangent)
{
    constexpr auto dx = real_type{1} / constants::sqrt_two;

    // Starting left of edge (-), headed down right (+,-)
    CheckedGeoTrackView geo{this->make_geo_track_view_interface()};
    geo = this->make_initializer({5 - dx, dx, 0}, {dx, -dx, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces up to a distance of 4 units away
    auto next = geo.find_next_step(from_cm(4.0));
    EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    // Move to boundary (-; +,-)
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("inner", this->volume_name(geo));

    // Reflect normal to surface (-; -,-)
    geo.set_dir(Real3{-dx, -dx, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("inner", this->volume_name(geo));

    // Crossing will *not* change volumes (-; -,-)
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    EXPECT_EQ("inner", this->volume_name(geo));

    // Find the next boundary and make sure that nearer distances aren't
    // accepted
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(10.0 * dx, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    EXPECT_TRUE(geo.is_on_boundary());
}

TEST_F(TwoBoxesTest, track)
{
    this->impl().test_detailed_tracking();
}

//---------------------------------------------------------------------------//
using ZnenvTest = GenericGeoParameterizedTest<GeantOrangeTest, ZnenvGeoTest>;

TEST_F(ZnenvTest, trace)
{
    this->impl().test_trace();
}

TEST_F(ZnenvTest, debug)
{
    auto geo = this->make_geo_track_view();
    geo = GeoTrackInitializer{{0.1, 0.0001, 0}, {1, 0, 0}};
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(
            R"json({"levels":[
{"dir":[1.0,0.0,0.0],"pos":[0.1,1e-4,0.0],"universe":"World","volume":{"canonical":"ZNTX","impl":"ZNTX","instance":"ZNTX_PV@1","local":2}},
{"dir":[1.0,0.0,0.0],"pos":[-1.66,1e-4,0.0],"universe":"ZNTX","volume":{"canonical":"ZN1","impl":"ZN1","instance":"ZN1_PV@1","local":2}},
{"dir":[1.0,0.0,0.0],"pos":[-1.66,-1.76,0.0],"universe":"ZN1","volume":{"canonical":"ZNSL","impl":"ZNSL","instance":"ZNSL_PV@0","local":1}},
{"dir":[1.0,0.0,0.0],"pos":[-1.66,-0.160,0.0],"universe":"ZNSL","volume":{"canonical":"ZNST","impl":"ZNST","instance":"ZNST_PV@0","local":1}},
{"dir":[1.0,0.0,0.0],"pos":[-0.0600,-0.160,0.0],"universe":"ZNST","volume":{"canonical":"ZNST","impl":"ZNST","instance":null,"local":5}}],
"surface":null})json",
            StringSimplifier{3}(to_json_string(geo.track_view())));
    }
    else
    {
        GTEST_SKIP() << "no gold results for this unit system";
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
