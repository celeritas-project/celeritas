//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/RecursiveSimplifier.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/RecursiveSimplifier.hh"

#include <iomanip>
#include <sstream>

#include "orange/surf/SurfaceIO.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class RecursiveSimplifierTest : public ::celeritas::test::Test
{
  protected:
};

TEST_F(RecursiveSimplifierTest, single)
{
    std::vector<std::string> result;

    auto append_to_result = [&](Sense sense, auto&& surf) {
        std::ostringstream os;
        os << to_char(sense) << std::setprecision(12) << surf;
        result.push_back(os.str());
    };
    RecursiveSimplifier simplify{append_to_result, 1e-6};

    // No simplification will occur
    simplify(Sense::inside, PlaneZ{1.5});
    // PlaneX{2} flipped
    simplify(Sense::outside, Plane{{-1, 0, 0}, -2.0});
    // CylY{{1,2,3}, 2.5} flipped and scaled
    simplify(Sense::inside,
             GeneralQuadric{{-2, 0, -2}, {0, 0, 0}, {4, 0, 12}, -2 * 3.75});

    static char const* const expected_result[]
        = {"-Plane: z=1.5", "-Plane: x=2", "+Cyl y: r=2.5 at x=1, z=3"};
    EXPECT_VEC_EQ(expected_result, result);
}

TEST_F(RecursiveSimplifierTest, variant)
{
    std::string senses;
    std::vector<std::string> stypes;
    std::vector<real_type> sdata;

    auto append_to_result = [&](Sense sense, auto&& surf) {
        senses.push_back(to_char(sense));
        stypes.push_back(to_cstring(surf.surface_type()));
        sdata.insert(sdata.end(), surf.data().begin(), surf.data().end());
    };
    RecursiveSimplifier simplify{append_to_result, 1e-6};

    std::vector<std::pair<Sense, VariantSurface>> const surfaces{
        {Sense::inside, SimpleQuadric{Plane{PlaneX{1}}}},
        {Sense::inside, Plane{{0, -1, 0}, 2}},
        {Sense::inside, Sphere{{0, 0, 0}, 1.0}},
        {Sense::outside, ConeZ{{1, 2, 3}, 0.6}},
        {Sense::outside, SimpleQuadric{{-2, -2, -2}, {0, 0, 0}, 9}},
        {Sense::outside, GeneralQuadric{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9}},
    };
    for (auto&& [sense, surf] : surfaces)
    {
        simplify(sense, surf);
    }

    EXPECT_EQ("-+-+-+", senses);
    static char const* const expected_stypes[]
        = {"px", "py", "sc", "kz", "sc", "gq"};
    EXPECT_VEC_EQ(expected_stypes, stypes);
    static double const expected_sdata[]
        = {1, -2, 1, 1, 2, 3, 0.36, 4.5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_VEC_SOFT_EQ(expected_sdata, sdata);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
