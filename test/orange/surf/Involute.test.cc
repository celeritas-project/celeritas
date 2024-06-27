//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Involute.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Involute.hh"

#include "celeritas_config.h"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
const double pi = 3.14159265358979323846;
namespace test
{
    TEST(InvoluteTest, construction)
    {
        Involute invo{{0, 0, 1}, 1.0, 0.5, 1.0, 0.5, 2.0};
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}), invo.origin());
        EXPECT_SOFT_EQ(1.0, invo.r_b());
        EXPECT_SOFT_EQ(0.5, invo.a());
        EXPECT_SOFT_EQ(1.0, invo.sign());
        EXPECT_SOFT_EQ(0.5, invo.tmin());
        EXPECT_SOFT_EQ(2.0, invo.tmax());

        auto invo2 = Involute::at_origin({1, 0, 1}, 2.0, 0.2, -1.0, 1.0, 3.0);
        EXPECT_VEC_SOFT_EQ((Real3{1, 0, 1}), invo2.origin());
        EXPECT_SOFT_EQ(2.0, invo2.r_b());
        EXPECT_SOFT_EQ(0.2, invo2.a());
        EXPECT_SOFT_EQ(-1.0, invo2.sign());
        EXPECT_SOFT_EQ(1.0, invo2.tmin());
        EXPECT_SOFT_EQ(3.0, invo2.tmax());

    }

    TEST(InvoluteTest, sense)
    {
        Involute invo{{0, 0, 1}, 1.0, 0.5*pi, 1.0, 1.732050808, 8.01536114};
        EXPECT_EQ(SignedSense::outside, invo.calc_sense({0, 0, 1}));
        EXPECT_EQ(SignedSense::outside, invo.calc_sense({0, 1.0, 1}));
        EXPECT_EQ(SignedSense::outside, invo.calc_sense({5, 9.0, 1}));

        EXPECT_EQ(SignedSense::inside, invo.calc_sense({-2.5, 0, 1}));
        EXPECT_EQ(SignedSense::inside, invo.calc_sense({5, 2.5, 1}));
        EXPECT_EQ(SignedSense::outside, invo.calc_sense({-5, -5, 1}));

        EXPECT_EQ(SignedSense::on,
                invo.calc_sense({
                    -2.628488755942478,
                    0.6579786205081288,
                    1,
                }));
        EXPECT_EQ(SignedSense::on,
                invo.calc_sense({
                    1.270156197058998,
                    7.514233446036193,
                    1,
                }));
        EXPECT_EQ(SignedSense::inside,
                invo.calc_sense({
                    -2.00000001,
                    6.061646482474157,
                    -0.99,
                }));
    }

} // namespace test
} // namespace celeritas