//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.test.cc
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"

#include "celeritas_test.hh"

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(AlgorithmsTest, minmax)
{
    EXPECT_EQ(1, celeritas::min<int>(1, 2));
    EXPECT_NE(0.2, celeritas::min<float>(0.1, 0.2));
    EXPECT_EQ(2, celeritas::max<int>(1, 2));
    EXPECT_NE(0.1, celeritas::max<float>(0.1, 0.2));
}

TEST(AlgorithmsTest, ipow)
{
    EXPECT_DOUBLE_EQ(1, celeritas::ipow<0>(0.0));
    EXPECT_EQ(123.456, celeritas::ipow<1>(123.456));
    EXPECT_EQ(8, (celeritas::ipow<3>(2)));
    EXPECT_FLOAT_EQ(0.001f, celeritas::ipow<3>(0.1f));
    EXPECT_EQ(1e4, celeritas::ipow<4>(10.0));
    EXPECT_TRUE((std::is_same<int, decltype(celeritas::ipow<4>(5))>::value));
}
