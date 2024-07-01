//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Involute.hh"

#include <cmath>
#include <iostream>

#include "celeritas_config.h"
#include "corecel/Constants.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
using constants::pi;
namespace test
{
TEST(InvoluteTest, construction)
{
    auto invo = Involute::involute({1, 0, 1}, 2.0, 0.2, -1.0, 1.0, 3.0);
    EXPECT_VEC_SOFT_EQ((Real3{1, 0, 1}), invo.origin());
    EXPECT_SOFT_EQ(2.0, invo.r_b());
    EXPECT_SOFT_EQ(0.2, invo.a());
    EXPECT_SOFT_EQ(-1.0, invo.sign());
    EXPECT_SOFT_EQ(1.0, invo.tmin());
    EXPECT_SOFT_EQ(3.0, invo.tmax());
}

TEST(InvoluteTest, sense)
{
    auto invo = Involute::involute(
        {1, 0, 1}, 1.0, 0.5 * pi, 1.0, 1.732050808, 8.01536114);
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({0, 0, 1}));
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({0, 1.0, 1}));
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({5, 9.0, 1}));

    EXPECT_EQ(SignedSense::inside, invo.calc_sense({-1.5, 0, 1}));
    EXPECT_EQ(SignedSense::inside, invo.calc_sense({6, 2.5, 1}));
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({-5, -5, 1}));

    auto invo2 = Involute::involute({1, 0, 1}, 1.0, 0.0, 1.0, 1.0, 8.01536114);
    EXPECT_EQ(SignedSense::on,
              invo2.calc_sense({
                  2.0,
                  0,
                  1,
              }));
}

TEST(Involute, normal)
{
    auto invo = Involute::involute({0, 0, 0}, 1.0, 0.5 * pi, 1.0, 0, 3.28);

    EXPECT_VEC_SOFT_EQ(
        make_unit_vector(Real3{0.875820953349345, 0.48263615454527, 0}),
        invo.calc_normal({-0.041518291154117815, 1.1189065160709872, 0}));
}

TEST(Involute, solve_intersect)
{
    // z coord of origin does not matter
    auto invo
        = Involute::involute({1, 1, 200}, 1.1, -0.5 * pi, 1.0, 0, 1.99 * pi);

    double u = 0.9933558377574788 * std::sin(1);
    double v = -0.11508335932330707 * std::sin(1);
    double w = std::cos(1);
    real_type convert = std::sqrt(ipow<2>(v) + ipow<2>(u) + ipow<2>(w))
                        / std::sqrt(ipow<2>(v) + ipow<2>(u));

    auto dist = invo.calc_intersections(
        Real3{-5.8653052986571326, -0.30468305643505367 + 1, 0},
        Real3{u, v, w},
        SurfaceState::on);

    EXPECT_SOFT_EQ(0.0, dist[0]);
    EXPECT_SOFT_EQ(8.2132658639260558, dist[1]);
    EXPECT_SOFT_EQ(9.167603472624553 * convert, dist[2]);
}

}  // namespace test
}  // namespace celeritas