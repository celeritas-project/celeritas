//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeant.test.cc
//---------------------------------------------------------------------------//
#include <string>

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/rasterize/SafetyImager.hh"

#include "OrangeGeoTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GeantOrangeTest : public OrangeGeoTestBase
{
  protected:
    void SetUp() final
    {
        ASSERT_EQ(CELERITAS_REAL_TYPE, CELERITAS_REAL_TYPE_DOUBLE)
            << "Converting Geant4 requires double-precision reals";
        this->build_gdml_geometry(this->geometry_basename() + ".gdml");
    }
    Constant unit_length() const final { return lengthunits::centimeter; }
};

//---------------------------------------------------------------------------//
class TestEm3GeantTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "testem3"; }
};

TEST_F(TestEm3GeantTest, trace)
{
    {
        auto result = this->track({-20.1}, {1, 0, 0});

        static char const* const expected_volumes[] = {
            "world", "pb",  "lar",  "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",   "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",  "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",   "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",  "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",   "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",  "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",   "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",  "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",   "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",  "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            0.1,  0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 4};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.050, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 2};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//
class TestEm3FlatGeantTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "testem3-flat"; }
};

TEST_F(TestEm3FlatGeantTest, trace)
{
    {
        auto result = this->track({-20.1}, {1, 0, 0});

        static char const* const expected_volumes[]
            = {"world",       "gap_0",  "absorber_0",  "gap_1",
               "absorber_1",  "gap_2",  "absorber_2",  "gap_3",
               "absorber_3",  "gap_4",  "absorber_4",  "gap_5",
               "absorber_5",  "gap_6",  "absorber_6",  "gap_7",
               "absorber_7",  "gap_8",  "absorber_8",  "gap_9",
               "absorber_9",  "gap_10", "absorber_10", "gap_11",
               "absorber_11", "gap_12", "absorber_12", "gap_13",
               "absorber_13", "gap_14", "absorber_14", "gap_15",
               "absorber_15", "gap_16", "absorber_16", "gap_17",
               "absorber_17", "gap_18", "absorber_18", "gap_19",
               "absorber_19", "gap_20", "absorber_20", "gap_21",
               "absorber_21", "gap_22", "absorber_22", "gap_23",
               "absorber_23", "gap_24", "absorber_24", "gap_25",
               "absorber_25", "gap_26", "absorber_26", "gap_27",
               "absorber_27", "gap_28", "absorber_28", "gap_29",
               "absorber_29", "gap_30", "absorber_30", "gap_31",
               "absorber_31", "gap_32", "absorber_32", "gap_33",
               "absorber_33", "gap_34", "absorber_34", "gap_35",
               "absorber_35", "gap_36", "absorber_36", "gap_37",
               "absorber_37", "gap_38", "absorber_38", "gap_39",
               "absorber_39", "gap_40", "absorber_40", "gap_41",
               "absorber_41", "gap_42", "absorber_42", "gap_43",
               "absorber_43", "gap_44", "absorber_44", "gap_45",
               "absorber_45", "gap_46", "absorber_46", "gap_47",
               "absorber_47", "gap_48", "absorber_48", "gap_49",
               "absorber_49", "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            0.1,  0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 4};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.05,  0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
               0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
               0.115, 0.285, 2};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//
class SimpleCmsGeantTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "simple-cms"; }
};

TEST_F(SimpleCmsGeantTest, trace)
{
    {
        auto result = this->track({-75, 0, 0}, {1, 0, 0});
        static char const* const expected_volumes[] = {"si_tracker",
                                                       "vacuum_tube",
                                                       "si_tracker",
                                                       "em_calorimeter",
                                                       "had_calorimeter",
                                                       "sc_solenoid",
                                                       "fe_muon_chambers",
                                                       "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {45, 60, 95, 50, 100, 100, 325, 300};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {22.5, 700, 47.5, 25, 50, 50, 162.5, 150};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        auto result = this->track({25, 0, 701}, {0, 0, -1});
        static char const* const expected_volumes[]
            = {"world", "vacuum_tube", "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 1400, 1300};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0.5, 5, 5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
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
class TransformedBoxGeantTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "transformed-box"; }
};

TEST_F(TransformedBoxGeantTest, trace)
{
    {
        auto result = this->track({0, 0, -25}, {0, 0, 1});
        static char const* const expected_volumes[] = {
            "world",
            "simple",
            "world",
            "enclosing",
            "tiny",
            "enclosing",
            "world",
            "simple",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            13,
            4,
            6,
            1.75,
            0.5,
            1.75,
            6,
            4,
            38,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        auto result = this->track({0.25, 0, -25}, {0., 0, 1});
        static char const* const expected_volumes[] = {
            "world",
            "simple",
            "world",
            "enclosing",
            "tiny",
            "enclosing",
            "world",
            "simple",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            12.834936490539,
            3.7320508075689,
            6.4330127018922,
            1.75,
            0.5,
            1.75,
            6,
            4,
            38,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        auto result = this->track({0, 0.25, -25}, {0, 0., 1});
        static char const* const expected_volumes[] = {
            "world",
            "simple",
            "world",
            "enclosing",
            "tiny",
            "enclosing",
            "world",
            "simple",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            13,
            4,
            6,
            1.75,
            0.5,
            1.75,
            6,
            4,
            38,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        auto result = this->track({0.01, -20, 0.20}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"world", "enclosing", "tiny", "enclosing", "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {18.5, 1.1250390198213, 0.75090449735279, 1.1240564828259, 48.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//
class ZnenvGeantTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "znenv"; }
};

TEST_F(ZnenvGeantTest, trace)
{
    static char const* const expected_mid_volumes[]
        = {"World", "ZNENV", "ZNST", "ZNST",  "ZNST", "ZNST", "ZNST",
           "ZNST",  "ZNST",  "ZNST", "ZNST",  "ZNST", "ZNST", "ZNST",
           "ZNST",  "ZNST",  "ZNST", "ZNST",  "ZNST", "ZNST", "ZNST",
           "ZNST",  "ZNST",  "ZNST", "ZNENV", "World"};
    static real_type const expected_mid_distances[]
        = {6.38, 0.1,  0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
           0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
           0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.1,  46.38};
    {
        auto result = this->track({-10, 0.0001, 0}, {1, 0, 0});
        EXPECT_VEC_EQ(expected_mid_volumes, result.volumes);
        EXPECT_VEC_SOFT_EQ(expected_mid_distances, result.distances);
    }
    {
        auto result = this->track({0.0001, -10, 0}, {0, 1, 0});
        EXPECT_VEC_EQ(expected_mid_volumes, result.volumes);
        EXPECT_VEC_SOFT_EQ(expected_mid_distances, result.distances);
    }
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
}  // namespace test
}  // namespace celeritas
