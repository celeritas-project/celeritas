//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Involute.hh"

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using constants::pi;
using Sign = Chirality;
Sign ccw = Chirality::left;
Sign cw = Chirality::right;
using Real2 = Involute::Real2;

//---------------------------------------------------------------------------//
TEST(InvoluteTest, construction)
{
    auto check_props = [](Involute const& invo) {
        EXPECT_VEC_SOFT_EQ((Real2{1, 0}), invo.origin());
        EXPECT_SOFT_EQ(2.0, invo.r_b());
        EXPECT_SOFT_EQ(pi - 0.2, invo.displacement_angle());
        EXPECT_EQ(invo.sign(), cw);
        EXPECT_SOFT_EQ(1.0, invo.tmin());
        EXPECT_SOFT_EQ(3.0, invo.tmax());
    };

    Involute invo{{1, 0}, 2.0, 0.2, cw, 1.0, 3.0};
    check_props(invo);

    {
        // Construct from raw data and check
        SCOPED_TRACE("reconstructed");
        Involute recon{invo.data()};
        check_props(recon);
    }
}

//! Python reference can be found in \file
TEST(InvoluteTest, sense)
{
    Involute invo{
        {1, 0}, 1.0, 0.5 * pi, ccw, 1.732050808, 1.732050808 + 1.99 * pi};
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({0, 0, 1}));
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({0, 1.0, 1}));
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({5, 9.0, 1}));

    EXPECT_EQ(SignedSense::inside, invo.calc_sense({-1.5, 0, 1}));
    EXPECT_EQ(SignedSense::inside, invo.calc_sense({6, 2.5, 1}));
    EXPECT_EQ(SignedSense::outside, invo.calc_sense({-5, -5, 1}));

    Involute invo2{
        {1, 0}, 1.0, 0.5 * pi, cw, 1.732050808, 1.732050808 + 1.99 * pi};
    EXPECT_EQ(SignedSense::outside,
              invo2.calc_sense({
                  3.628488755942478 + 1e-5,
                  0.6579786205081288 + 1e-5,
                  1,
              }));
    EXPECT_EQ(SignedSense::inside,
              invo2.calc_sense({
                  3.628488755942478 - 1e-5,
                  0.6579786205081288 - 1e-5,
                  1,
              }));
    Involute invo3{
        {1, 0}, 2.0, 0.5 * pi, ccw, 1.732050808, 1.732050808 + 1.99 * pi};
    EXPECT_EQ(SignedSense::outside, invo3.calc_sense({4.999, 0, 1}));
    EXPECT_EQ(SignedSense::inside, invo3.calc_sense({5.001, 0, 1}));
}

//! Python reference can be found in \file
TEST(Involute, normal)
{
    Involute invo{{0, 0}, 1.0, 0.5 * pi, cw, 0, 3.14};
    EXPECT_VEC_SOFT_EQ(
        make_unit_vector(Real3{-0.968457782598019, 0.249177694277252, 0}),
        invo.calc_normal({0.005289930339633125, 1.0312084690733585, 0}));

    Involute invo2{{0, 0}, 1.0, 0.5 * pi, ccw, 0, 3.14};
    EXPECT_VEC_SOFT_EQ(
        make_unit_vector(Real3{0.968457782598019, 0.249177694277252, 0}),
        invo2.calc_normal({-0.005289930339633125, 1.0312084690733585, 0}));

    Involute invo3{{0, 0}, 2.0, 0, ccw, 0, 3.14};
    EXPECT_VEC_SOFT_EQ(make_unit_vector(Real3{0, -1, 0}),
                       invo3.calc_normal({2, 0, 0}));
}

TEST(Involute, solve_intersect)
{
    // Only testing call of solver, more extensive test in InvoluteSolver

    // CCW Involute Test
    {
        Involute invo{{1, 1}, 1.1, 1.5 * pi, ccw, 0, 1.99 * pi};

        real_type u = 0.9933558377574788 * std::sin(1);
        real_type v = -0.11508335932330707 * std::sin(1);
        real_type w = std::cos(1);
        real_type convert = 1 / hypot(v, u);

        auto dist = invo.calc_intersections(
            Real3{-5.8653259986571326, -0.30468105643505367 + 1, 0},
            Real3{u, v, w},
            SurfaceState::on);

        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_EQ(6.9112457587355429 * convert, dist[0]);
            EXPECT_SOFT_EQ(9.1676238065759748 * convert, dist[1]);
            EXPECT_SOFT_EQ(2.0792209373995243e-05 * convert, dist[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(6.9112457587355429 * convert, dist[0]);
            EXPECT_SOFT_EQ(9.1676238065759748 * convert, dist[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist[2]);
        }
    }

    // CW Involute Test
    {
        Involute invo{{0.0, 0.0}, 0.5, 0.4 * pi, cw, 2, 4};

        real_type u = 0.894427191 * 0.5;
        real_type v = -0.4472135955 * 0.5;
        real_type w = std::sqrt(3) * 0.25;
        real_type convert = 2;

        auto dist = invo.calc_intersections(
            Real3{-4.0, 2.0, 0}, Real3{u, v, w}, SurfaceState::off);

        EXPECT_SOFT_EQ(6.0371012194546871 * convert, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Standard Involute
    {
        Involute invo{{0, 0}, 1.0, 0.0, ccw, 0.5, 4};

        real_type x = 0;
        real_type y = -2;
        real_type u = 1;
        real_type v = 0;

        auto dist = invo.calc_intersections(
            Real3{x, y, 0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }
}

}  // namespace test
}  // namespace celeritas
