//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/SplineInterpolator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/SplineInterpolator.hh"

#include <vector>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(SplineInterpolateTest, all)
{
    std::vector<real_type> x{0, 1, 2, 3, 4};
    std::vector<real_type> y{0, 2, 1, 2, 0};
    {
        // Natural boundary conditions
        std::vector<real_type> ddy{0, -6, 6, -6, 0};

        SplineInterpolator interpolate({x[0], y[0], ddy[0]},
                                       {x[1], y[1], ddy[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.299, interpolate(0.1));
        EXPECT_SOFT_EQ(1.375, interpolate(0.5));
        EXPECT_SOFT_EQ(1.971, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));
    }
    {
        // Not-a-knot boundary conditions
        std::vector<real_type> ddy{-10.5, -3, 4.5, -3, -10.5};

        SplineInterpolator interpolate({x[0], y[0], ddy[0]},
                                       {x[1], y[1], ddy[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.54875, interpolate(0.1));
        EXPECT_SOFT_EQ(1.84375, interpolate(0.5));
        EXPECT_SOFT_EQ(2.05875, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));

        interpolate
            = SplineInterpolator({x[1], y[1], ddy[1]}, {x[2], y[2], ddy[2]});
        EXPECT_EQ(2, interpolate(1));
        EXPECT_SOFT_EQ(1.40625, interpolate(1.5));
        EXPECT_SOFT_EQ(1.00000224875, interpolate(1.999));
        EXPECT_EQ(1, interpolate(2));
    }
    {
        // Geant4's not-a-knot boundary conditions
        std::vector<real_type> ddy{-4.3125, 0, 4.3125, -3, -10.3125};

        SplineInterpolator interpolate({x[0], y[0], ddy[0]},
                                       {x[1], y[1], ddy[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.32290625, interpolate(0.1));
        EXPECT_SOFT_EQ(1.26953125, interpolate(0.5));
        EXPECT_SOFT_EQ(1.87115625, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));

        interpolate
            = SplineInterpolator({x[1], y[1], ddy[1]}, {x[2], y[2], ddy[2]});
        EXPECT_EQ(2, interpolate(1));
        EXPECT_SOFT_EQ(1.23046875, interpolate(1.5));
        EXPECT_SOFT_EQ(0.99956465553125, interpolate(1.999));
        EXPECT_EQ(1, interpolate(2));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
