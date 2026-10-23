//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/inp/IO.test.cc
//---------------------------------------------------------------------------//
#include <sstream>
#include <vector>

#include "corecel/cont/Range.hh"
#include "orange/inp/IO.json.hh"
#include "orange/inp/Import.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace inp
{
namespace test
{
//---------------------------------------------------------------------------//
// UniverseAlignmentIO tests
//---------------------------------------------------------------------------//
TEST(UniverseAlignmentIOTest, round_trip)
{
    // Lambda for flattening SquareMatrixReal3
    auto flatten = [](SquareMatrixReal3 const& mat) {
        std::vector<real_type> out;
        for (auto i : range(3))
        {
            for (auto j : range(3))
            {
                out.push_back(mat[i][j]);
            }
        }
        return out;
    };

    // Create rotations matrices that are quarter turns around z and x
    // clang-format off
    SquareMatrixReal3 const rot_x{
        Real3{1, 0, 0},
        Real3{0, 0, 1},
        Real3{0, -1, 0}};
    SquareMatrixReal3 const rot_z{
        Real3{0, -1, 0},
        Real3{1, 0, 0},
        Real3{0, 0, 1}};
    // clang-format on

    // Set rotation matrices to LV names
    OrangeGeoFromGeant opts;
    opts.universe_alignment["lv_0"] = rot_x;
    opts.universe_alignment["lv_1"] = rot_z;

    // Write and read back the options
    std::stringstream ss;
    ss << opts;
    OrangeGeoFromGeant rt;
    ss >> rt;

    // Test that we can roundtrip
    ASSERT_EQ(2, rt.universe_alignment.size());
    EXPECT_VEC_EQ(flatten(rot_x), flatten(rt.universe_alignment.at("lv_0")));
    EXPECT_VEC_EQ(flatten(rot_z), flatten(rt.universe_alignment.at("lv_1")));
}

TEST(UniverseAlignmentIOTest, errors)
{
    // Lambda for creating OrangeGeoFromGeant with a supplied
    // universe_alignment value
    auto from_alignment = [](std::string const& value) {
        OrangeGeoFromGeant result;
        std::istringstream{R"json({"_format": "g4org-options", )json"
                           R"json("universe_alignment": )json"
                           + value + "}"}
            >> result;
        return result;
    };

    // Not a matrix
    EXPECT_THROW(from_alignment(R"({"a": 5})"), RuntimeError);
    // Wrong matrix shape
    EXPECT_THROW(from_alignment(R"({"a": [[1,0,0],[0,1,0]]})"), RuntimeError);
    EXPECT_THROW(from_alignment(R"({"a": [[1,0],[0,1],[0,0]]})"), RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace inp
}  // namespace celeritas
