//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/SurfaceInserter.test.cc
//---------------------------------------------------------------------------//
#include "orange/construct/SurfaceInserter.hh"

#include <fstream>

#include "celeritas_config.h"
#include "orange/construct/SurfaceInput.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/GeneralQuadric.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/Sphere.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_JSON
#    include "orange/construct/SurfaceInputIO.json.hh"
#endif

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceInserterTest : public celeritas_test::Test
{
  protected:
    ::celeritas::HostVal<SurfaceData> surface_data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SurfaceInserterTest, manual)
{
    SurfaceInserter insert(&surface_data_);
    EXPECT_EQ(SurfaceId{0}, insert(PlaneX(1)));
    EXPECT_EQ(SurfaceId{1}, insert(CCylX(2)));
    EXPECT_EQ(SurfaceId{2}, insert(Sphere({1, 2, 3}, 4)));
    EXPECT_EQ(SurfaceId{3},
              insert(GeneralQuadric({0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9)));

    EXPECT_EQ(4, surface_data_.types.size());
    EXPECT_EQ(4, surface_data_.offsets.size());

    const double expected_reals[]
        = {1, 4, 1, 2, 3, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_VEC_SOFT_EQ(expected_reals,
                       surface_data_.reals[AllItems<real_type>{}]);
}

TEST_F(SurfaceInserterTest, from_input)
{
    SurfaceInput input;
    input.types = {SurfaceType::px, SurfaceType::s};
    input.data  = {1.25, 4, 0, 1, 2};
    input.sizes = {1, 4};

    SurfaceInserter insert(&surface_data_);

    // Initial insert
    auto surface_range = insert(input);
    EXPECT_EQ(SurfaceId{0}, *surface_range.begin());
    EXPECT_EQ(2, surface_range.size());

    // Insert again
    surface_range = insert(input);
    EXPECT_EQ(SurfaceId{2}, *surface_range.begin());
    EXPECT_EQ(2, surface_range.size());
}

TEST_F(SurfaceInserterTest, from_json)
{
    SurfaceInserter insert(&surface_data_);
    std::ifstream   infile(
        this->test_data_path("orange", "five-volumes.org.json"));

#if !CELERITAS_USE_JSON
    GTEST_SKIP() << "JSON is not enabled";
#else
    auto        full_inp = nlohmann::json::parse(infile);
    const auto& surfaces = full_inp["universes"][0]["surfaces"];

    auto surface_range = insert(surfaces.get<SurfaceInput>());
    EXPECT_EQ(SurfaceId{0}, *surface_range.begin());
    EXPECT_EQ(12, surface_range.size());
#endif
}
