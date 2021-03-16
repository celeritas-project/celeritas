//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Interpolator.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/Interpolator.hh"

#include "celeritas_test.hh"

using celeritas::Interp;
using celeritas::Interpolator;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST(InterpolateTest, lin_lin)
{
    using Interpolator_t = Interpolator<Interp::linear, Interp::linear, double>;
    // Interpolate in x between (x,y) = (.1, 10) and (1.1,20)
    Interpolator_t interp({0.1, 10}, {1.1, 20});

    EXPECT_DOUBLE_EQ(10., interp(0.1));
    EXPECT_DOUBLE_EQ(15., interp(0.6));
    EXPECT_DOUBLE_EQ(20., interp(1.1));
}

//---------------------------------------------------------------------------//
TEST(InterpolateTest, lin_log)
{
    using Interpolator_t = Interpolator<Interp::log, Interp::linear, double>;
    Interpolator_t interp({1, 0.1}, {100, 1.1});
    EXPECT_DOUBLE_EQ(0.1, interp(1.));
    EXPECT_DOUBLE_EQ(0.6, interp(10.));
    EXPECT_DOUBLE_EQ(1.1, interp(100.));
}

//---------------------------------------------------------------------------//
TEST(InterpolateTest, log_lin)
{
    using Interpolator_t = Interpolator<Interp::linear, Interp::log, double>;
    Interpolator_t interp({0.1, 1}, {1.1, 100});
    EXPECT_DOUBLE_EQ(1., interp(0.1));
    EXPECT_DOUBLE_EQ(10., interp(0.6));
    EXPECT_DOUBLE_EQ(100., interp(1.1));
}

//---------------------------------------------------------------------------//
TEST(InterpolateTest, log_log)
{
    using Interpolator_t = Interpolator<Interp::log, Interp::log, double>;
    Interpolator_t interp({1, 1e5}, {100, 1e7});
    EXPECT_DOUBLE_EQ(1e5, interp(1.));
    EXPECT_DOUBLE_EQ(1e6, interp(10.));
    EXPECT_SOFT_EQ(1e7, interp(100.));
}
