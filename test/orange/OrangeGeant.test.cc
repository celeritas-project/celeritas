//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeant.test.cc
//---------------------------------------------------------------------------//
#include <string>

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/detail/LengthUnits.hh"

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
    real_type unit_length() const final { return lengthunits::centimeter; }
};

//---------------------------------------------------------------------------//
class FourLevelsTest : public GeantOrangeTest
{
    std::string geometry_basename() const override { return "four-levels"; }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, accessors)
{
    auto const& geom = *this->geometry();
    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-24., -24., -24.}), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{24., 24., 24.}), to_cm(bbox.upper()));

    ASSERT_EQ(14, geom.num_volumes());
    EXPECT_EQ("World", geom.id_to_label(VolumeId{9}).name);
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{11}).name);
    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{12}).name);
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{13}).name);
    EXPECT_EQ(Label("World", "0xdeadbeef"), geom.id_to_label(VolumeId{13}));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, consecutive_compute)
{
    auto geo = this->make_geo_track_view({-9, -10, -10}, {1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape2", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    auto next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    // Find safety from a freshly initialized state
    geo = {from_cm({-9, -10, -10}), {1, 0, 0}};
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, detailed_track)
{
    {
        SCOPED_TRACE("rightward along corner");
        auto geo = this->make_geo_track_view({-10, -10, -10}, {1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape2", this->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        auto next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        geo.move_internal(from_cm(3.5));
        EXPECT_FALSE(geo.is_on_boundary());

        // Find one a bit further, then cross it
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(1.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_EQ("Shape2", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("Shape1", this->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Find the next boundary and make sure that nearer distances aren't
        // accepted
        next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        next = geo.find_next_step(from_cm(0.5));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("inside out");
        auto geo = this->make_geo_track_view({-23.5, 6.5, 6.5}, {-1, 0, 0});
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ("World", this->volume_name(geo));

        auto next = geo.find_next_step(from_cm(2));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_TRUE(geo.is_outside());
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, reentrant_boundary)
{
    auto geo = this->make_geo_track_view({15.5, 10, 10}, {-1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape1", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces: we should hit the outside of the sphere Shape2
    auto next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
    // Move to the boundary but scatter perpendicularly, away from the sphere
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, 1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("Shape1", this->volume_name(geo));

    // Scatter back toward the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(6, to_cm(next.distance));
    geo.set_dir({-1, 0, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("Shape1", this->volume_name(geo));

    // Move to the sphere boundary then scatter still into the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(1e-13, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, -1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ("Shape2", this->volume_name(geo));

    EXPECT_TRUE(geo.is_on_boundary());

    // Travel nearly tangent to the right edge of the sphere, then scatter to
    // still outside
    next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(9.9794624025613538e-07, to_cm(next.distance));
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({1, 0, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ("Shape1", this->volume_name(geo));
    EXPECT_TRUE(geo.is_on_boundary());
    next = geo.find_next_step(from_cm(10.0));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, tracking)
{
    constexpr real_type safety_tol{1e-10};
    {
        SCOPED_TRACE("Rightward");
        auto result = this->track({-10, -10, -10}, {1, 0, 0});

        static char const* const expected_volumes[] = {"Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World",
                                                       "Envelope",
                                                       "Shape1",
                                                       "Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {5, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {2.5, 0.5, 0.5, 3, 0.5, 0.5, 5, 0.5, 0.5, 3.5};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("From just inside outside edge");
        auto result = this->track({-24 + 0.001, 10., 10.}, {1, 0, 0});

        static char const* const expected_volumes[] = {"World",
                                                       "Envelope",
                                                       "Shape1",
                                                       "Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World",
                                                       "Envelope",
                                                       "Shape1",
                                                       "Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {7 - 0.001, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {3.4995, 0.5, 0.5, 5, 0.5, 0.5, 3, 0.5, 0.5, 5, 0.5, 0.5, 3.5};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = this->track({-10, 10, 10}, {0, 1, 0});

        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 2, 6};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 0.5, 1, 3};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = this->track({-10, 10, 10}, {0, 0, 1});

        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 3, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 0.5, 1.5, 2.5};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, safety)
{
    auto geo = this->make_geo_track_view();
    std::vector<real_type> safeties;
    std::vector<real_type> lim_safeties;

    for (auto i : range(11))
    {
        real_type r = from_cm(2.0 * i + 0.1);
        geo = {{r, r, r}, {1, 0, 0}};
        if (!geo.is_outside())
        {
            geo.find_next_step();
            safeties.push_back(to_cm(geo.find_safety()));
            lim_safeties.push_back(to_cm(geo.find_safety(from_cm(1.5))));
        }
    }

    static double const expected_safeties[] = {
        2.9,
        0.9,
        0.1,
        1.7549981495186,
        1.7091034656191,
        4.8267949192431,
        1.3626933041054,
        1.9,
        0.1,
        1.1,
        3.1,
    };
    EXPECT_VEC_SOFT_EQ(expected_safeties, safeties);

    static double const expected_lim_safeties[] = {
        2.9,
        0.9,
        0.1,
        1.7549981495186,
        1.7091034656191,
        4.8267949192431,
        1.3626933041054,
        1.9,
        0.1,
        1.1,
        3.1,
    };
    EXPECT_VEC_SOFT_EQ(expected_lim_safeties, lim_safeties);
}

//---------------------------------------------------------------------------//
class TestEm3GeantTest : public GeantOrangeTest
{
    std::string geometry_basename() const final { return "testem3"; }
};

TEST_F(TestEm3GeantTest, trace)
{
    {
        auto result = this->track({-20.1}, {1, 0, 0});

        static char const* const expected_volumes[]
            = {"World",  "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb",
               "G4_lAr", "G4_Pb", "G4_lAr", "G4_Pb", "G4_lAr", "World"};
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
}  // namespace test
}  // namespace celeritas
